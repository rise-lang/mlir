//
// Created by martin on 26/11/2019.
//

#include "mlir/Conversion/RiseToImperative/ConvertRiseToImperative.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>
#include <mlir/EDSC/Builders.h>

using namespace mlir;
using namespace mlir::rise;
using namespace mlir::edsc;

namespace {
struct ConvertRiseToImperativePass
    : public RiseToImperativeBase<ConvertRiseToImperativePass> {
  void runOnFunction() override;
};

// Codegen phase 1
void lowerAndStore(Value expr, Value out, PatternRewriter &rewriter);
mlir::Value lower(mlir::Value expr, Block::iterator contLocation,
                  PatternRewriter &rewriter);

void lowerAndStoreReduceSeq(NatAttr n, DataTypeAttr s, DataTypeAttr t,
                            Value reductionFun, Value initializer, Value array,
                            Value out, Location loc, PatternRewriter &rewriter,
                            StringRef loweringTarget = "scf");
void lowerAndStoreMapSeq(NatAttr n, DataTypeAttr s, DataTypeAttr t, Value f,
                         Value array, Value out, Location loc,
                         PatternRewriter &rewriter,
                         StringRef loweringTarget = "scf");
void lowerAndStoreFst(Value tuple, Value out, Location loc,
                      PatternRewriter &rewriter);
void lowerAndStoreSnd(Value tuple, Value out, Location loc,
                      PatternRewriter &rewriter);
void lowerAndStoreSplit(NatAttr n, NatAttr m, DataTypeAttr t, Value array,
                        Value out, Location loc, PatternRewriter &rewriter);
void lowerAndStoreJoin(NatAttr n, NatAttr m, DataTypeAttr t, Value array,
                       Value out, Location loc, PatternRewriter &rewriter);

mlir::Value lowerLiteral(Value literalValue, Location loc,
                         PatternRewriter &rewriter);
mlir::Value lowerMapSeq(NatAttr n, Value val, Location loc,
                        PatternRewriter &rewriter);
mlir::Value lowerMap(NatAttr n, DataTypeAttr s, DataTypeAttr t, LambdaOp mapFun,
                     Value array, Type type, Location loc,
                     PatternRewriter &rewriter);
mlir::Value lowerZip(Value lhs, Value rhs, Location loc,
                     PatternRewriter &rewriter);
mlir::Value lowerFst(Value tuple, Location loc, PatternRewriter &rewriter);
mlir::Value lowerSnd(Value tuple, Location loc, PatternRewriter &rewriter);
mlir::Value lowerSplit(NatAttr n, NatAttr m, DataTypeAttr t, Type type,
                       Value array, Location loc, PatternRewriter &rewriter);
mlir::Value lowerJoin(NatAttr n, NatAttr m, DataTypeAttr t, Type type,
                      Value array, Location loc, PatternRewriter &rewriter);
mlir::Value lowerTranspose(NatAttr n, NatAttr m, DataTypeAttr t, Type type,
                           Value array, Location loc,
                           PatternRewriter &rewriter);
mlir::Value lowerSlide(NatAttr n, NatAttr sz, NatAttr sp, DataTypeAttr t,
                       Type type, Value array, Location loc,
                       PatternRewriter &rewriter);
mlir::Value lowerPad(NatAttr n, NatAttr l, NatAttr r, DataTypeAttr t, Type type,
                     Value array, Location loc, PatternRewriter &rewriter);

// Codegen phase 2
void lowerAssign(AssignOp assignOp, PatternRewriter &rewriter);
Value resolveIndexing(Value val, SmallVector<OutputPathType, 10> path,
                      PatternRewriter &rewriter);
SmallVector<OutputPathType, 10>
resolveStoreIndexing(Value storeLocation, Value val,
                     SmallVector<OutputPathType, 10> path,
                     PatternRewriter &rewriter);
Value generateWriteAccess(SmallVector<OutputPathType, 10> path, Value accessVal,
                          PatternRewriter &rewriter);
void generateReadAccess(SmallVector<OutputPathType, 10> path, Value storeVal,
                        Value storeLoc, PatternRewriter &rewriter);

// utils
void Substitute(LambdaOp lambda, llvm::SmallVector<Value, 10> args);
void printPath(SmallVector<OutputPathType, 10> path,
               StringRef additionalInfo = "");
void printUses(Value val);

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

struct RiseToImperativePattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;
  LogicalResult match(FuncOp funcOp) const override;
  void rewrite(FuncOp funcOp, PatternRewriter &rewriter) const override;
};

LogicalResult RiseToImperativePattern::match(FuncOp funcOp) const {
  bool riseInside = false;

  if (funcOp.isExternal())
    return failure();

  // Only unlowered rise programs contain RiseInOps
  funcOp.walk([&](Operation *op) {
    if (isa<InOp>(op))
      riseInside = true;
  });

  if (riseInside) {
    return success();
  } else {
    return failure();
  }
}

void RiseToImperativePattern::rewrite(FuncOp funcOp,
                                      PatternRewriter &rewriter) const {
  Block &block = funcOp.getBody().front();
  funcOp.walk([&](Operation *op) {
    if (InOp inOp = dyn_cast<InOp>(op)) {
      rewriter.setInsertionPointAfter(inOp);
    }
  });

  // Start at the back and find the rise.out op
  auto _outOp = std::find_if(block.rbegin(), block.rend(),
                             [](auto &op) { return isa<OutOp>(op); });
  if (_outOp == block.rend()) {
    emitError(funcOp.getLoc()) << "Could not find rise.out operation!";
    return;
  }
  OutOp outOp = dyn_cast<OutOp>(*_outOp);

  // insert cast of out to fit our type system
  Value out = rewriter
                  .create<CastOp>(outOp.getLoc(), outOp.getOperand(1).getType(),
                                  outOp.getOperand(0))
                  .getResult();

  // first lowering phase:
  if (ApplyOp apply = dyn_cast<ApplyOp>(outOp.getOperand(1).getDefiningOp())) {
    lowerAndStore(apply.getResult(), out, rewriter);
  } else {
    emitError(outOp.getLoc())
        << "Result of rise.out has to be the result of a rise.apply op!";
    return;
  }

  // Cleanup of leftover ops from lowerAndStore
  SmallVector<Operation *, 10> leftoverOps = {};
  funcOp.walk([&leftoverOps](Operation *inst) {
    if (!inst)
      return;
    if (isa<ApplyOp>(inst) || isa<LambdaOp>(inst) || isa<MapSeqOp>(inst) ||
        isa<MapParOp>(inst) || isa<MapOp>(inst) || isa<ReduceSeqOp>(inst) ||
        isa<OutOp>(inst) || isa<LiteralOp>(inst) || isa<TransposeOp>(inst) ||
        isa<SplitOp>(inst) || isa<JoinOp>(inst) || isa<SlideOp>(inst) ||
        isa<PadOp>(inst)) {
      if (!inst->getParentOfType<LambdaOp>()) {
        leftoverOps.push_back(inst);
      }
    }
    return;
  });

  while (!leftoverOps.empty()) {
    auto op = leftoverOps.pop_back_val();
    op->dropAllUses();
    op->dropAllReferences();
    rewriter.eraseOp(op);
  }

  emitRemark(funcOp.getLoc())
      << "lowerAndStore finished. Starting resolving indicees.";

  SmallVector<rise::AssignOp, 10> assignOps;
  funcOp.walk([&assignOps](Operation *inst) {
    if (inst && isa<AssignOp>(inst)) {
      assignOps.push_back(cast<AssignOp>(inst));
    }
    return;
  });

  // second lowering phase:
  bool doCodegen = true;
  SmallVector<Operation *, 10> erasureList = {};
  if (doCodegen) {
    for (rise::AssignOp assign : assignOps) {
      lowerAssign(assign, rewriter);
    }
    emitRemark(funcOp.getLoc())
        << "Resolving indicees finished. Starting Cleanup.";
  }

  //   cleanup:
  //   erase intermediate operations.
  if (doCodegen) {
    funcOp.walk([&](Operation *inst) {
      if (inst->getDialect() &&
          inst->getDialect()->getNamespace().equals("rise")) {
        if (!(inst->getParentOfType<LambdaOp>() ||
              inst->getParentOfType<EmbedOp>())) {
          erasureList.push_back(inst);
        }
      }
      return;
    });
  }

  while (!erasureList.empty()) {
    auto op = erasureList.pop_back_val();
    op->dropAllUses();
    op->dropAllReferences();
    rewriter.eraseOp(op);
  }

  return;
}

//===----------------------------------------------------------------------===//
// First part of the lowering process
//===----------------------------------------------------------------------===//

void lowerAndStore(Value expr, Value out, PatternRewriter &rewriter) {
  if (out.getType().getDialect().getNamespace() !=
      RiseDialect::getDialectNamespace()) {
    emitError(out.getLoc()) << "type of out is wrong! Dumping it";
    out.getType().dump();
    return;
  }
  OpBuilder::InsertPoint savedInsertionPoint = rewriter.saveInsertionPoint();

  if (ApplyOp apply = dyn_cast<ApplyOp>(expr.getDefiningOp())) {
    Operation *appliedFun = apply.getOperand(0).getDefiningOp();
    Location loc = apply.getLoc();

    // lowering to a specific loop depending on the lowering target dialect
    StringRef loweringTarget = [&] {
      if (StringAttr loweringTargetAttr =
              appliedFun->getAttrOfType<StringAttr>("to")) {
        return loweringTargetAttr.getValue();
      } else {
        return StringRef("scf"); // default lowering target
      }
    }();

    // Dispatch to the correct lowerAndStore
    if (ReduceSeqOp reduceSeqOp = dyn_cast<ReduceSeqOp>(appliedFun)) {
      emitRemark(reduceSeqOp.getLoc()) << "lowerAndStore of ReduceSeq";
      lowerAndStoreReduceSeq(reduceSeqOp.nAttr(), reduceSeqOp.sAttr(),
                             reduceSeqOp.tAttr(), apply.getOperand(1),
                             apply.getOperand(2), apply.getOperand(3), out,
                             reduceSeqOp.getLoc(), rewriter, loweringTarget);
      return;
    } else if (MapSeqOp mapSeqOp = dyn_cast<MapSeqOp>(appliedFun)) {
      emitRemark(mapSeqOp.getLoc()) << "lowerAndStore of MapSeq";
      mapSeqOp.getParentOfType<FuncOp>().dump();
      lowerAndStoreMapSeq(mapSeqOp.nAttr(), mapSeqOp.sAttr(), mapSeqOp.tAttr(),
                          apply.getOperand(1), apply.getOperand(2), out,
                          mapSeqOp.getLoc(), rewriter, loweringTarget);
      return;
    } else if (MapParOp mapParOp = dyn_cast<MapParOp>(appliedFun)) {
      emitRemark(mapParOp.getLoc())
          << "lowerAndStore of MapPar, currently not generating parallel code!";
      lowerAndStoreMapSeq(mapParOp.nAttr(), mapParOp.sAttr(), mapParOp.tAttr(),
                          apply.getOperand(1), apply.getOperand(2), out,
                          mapParOp.getLoc(), rewriter, loweringTarget);
      return;
    } else if (MapOp mapOp = dyn_cast<MapOp>(appliedFun)) {
      emitError(appliedFun->getLoc()) << "lowerAndStore of Map unsupported!";
      return;
    } else if (FstOp fstOp = dyn_cast<FstOp>(appliedFun)) {
      emitRemark(fstOp.getLoc()) << "lowerAndStore of Fst";
      lowerAndStoreFst(apply.getOperand(1), out, fstOp.getLoc(), rewriter);
      return;
    } else if (SndOp sndOp = dyn_cast<SndOp>(appliedFun)) {
      emitRemark(sndOp.getLoc()) << "lowerAndStore of Snd";
      lowerAndStoreSnd(apply.getOperand(1), out, sndOp.getLoc(), rewriter);
      return;
    } else if (SplitOp splitOp = dyn_cast<SplitOp>(appliedFun)) {
      emitRemark(splitOp.getLoc()) << "lowerAndStore of Split";
      lowerAndStoreSplit(splitOp.nAttr(), splitOp.mAttr(), splitOp.tAttr(),
                         apply.getOperand(1), out, splitOp.getLoc(), rewriter);
      return;
    } else if (JoinOp joinOp = dyn_cast<JoinOp>(appliedFun)) {
      emitRemark(joinOp.getLoc()) << "AccT of Join";
      lowerAndStoreJoin(joinOp.nAttr(), joinOp.mAttr(), joinOp.tAttr(),
                        apply.getOperand(1), out, joinOp.getLoc(), rewriter);
      return;
    } else if (LambdaOp lambdaOp = dyn_cast<LambdaOp>(appliedFun)) {
      emitRemark(appliedFun->getLoc()) << "lowerAndStore of Lambda";
      SmallVector<Value, 10> args = SmallVector<Value, 10>();
      for (int i = apply.getNumOperands() - 1; i > 0; i--) {
        args.push_back(apply.getOperand(i));
      }
      Substitute(lambdaOp, args);

      // Find return in Lambda Region to start new lowerAndStore
      rise::ReturnOp returnOp =
          dyn_cast<rise::ReturnOp>(lambdaOp.region().front().getTerminator());
      lowerAndStore(returnOp.getOperand(0), out, rewriter);
      return;
    }
    emitRemark(appliedFun->getLoc())
        << "Can't lower the application of op: " << appliedFun->getName();
    return;
  }

  // We lower the operand of a rise::returnOp
  if (!expr.isa<OpResult>()) {
    emitError(expr.getLoc()) << "Directly returning an argument in a "
                                "rise.return is not supported in lowering to "
                                "imperative currently";
    return;
  }
  if (ApplyOp apply = dyn_cast<ApplyOp>(expr.getDefiningOp())) {
    lowerAndStore(apply.getResult(), out, rewriter);
    return;
  }
  if (EmbedOp embedOp = dyn_cast<EmbedOp>(expr.getDefiningOp())) {
    emitRemark(expr.getLoc()) << "lowerAndStore of EmbedOp. Copy "
                                 "operations from this block to result.";
    assert(
        embedOp.getNumOperands() ==
            embedOp.region().front().getNumArguments() &&
        "Embed has to have the same number of operands and block arguments!");

    // Translating all operands first
    for (int i = 0; i < embedOp.getOperands().size(); i++) {
      auto operand = embedOp.getOperand(i);
      auto operandCont = lower(operand, rewriter.getInsertionPoint(), rewriter);
      embedOp.setOperand(i, operandCont);
    }
    auto newEmbed = rewriter.clone(*embedOp.getOperation());
    auto assignment = rewriter.create<AssignOp>(newEmbed->getLoc(),
                                                newEmbed->getResult(0), out);
    return;
  }
  emitError(expr.getLoc()) << "lowerAndStore of a rise.return went wrong!";
}

mlir::Value lower(mlir::Value expr, Block::iterator contLocation,
                  PatternRewriter &rewriter) {
  Location loc = expr.getLoc();
  auto oldInsertPoint = rewriter.saveInsertionPoint();

  if (!expr.isa<OpResult>()) {
    emitRemark(expr.getLoc())
        << "cannot perform continuation for BlockArg, leaving as is.";
    return expr;
  }
  if (LiteralOp literalOp = dyn_cast<LiteralOp>(expr.getDefiningOp())) {
    emitRemark(literalOp.getLoc()) << "lower of Literal";
    Value lowered =
        lowerLiteral(literalOp.getResult(), literalOp.getLoc(), rewriter);
    rewriter.restoreInsertionPoint(oldInsertPoint);
    return lowered;
  } else if (ApplyOp apply = dyn_cast<ApplyOp>(expr.getDefiningOp())) {
    if (ZipOp zipOp = dyn_cast<ZipOp>(apply.fun().getDefiningOp())) {
      emitRemark(expr.getLoc()) << "lower of Applied Zip";
      return lowerZip(apply.getOperand(1), apply.getOperand(2), zipOp.getLoc(),
                      rewriter);
    } else if (FstOp fst = dyn_cast<FstOp>(apply.fun().getDefiningOp())) {
      emitRemark(expr.getLoc()) << "lower of Applied Fst";
      return lowerFst(apply.getOperand(1), fst.getLoc(), rewriter);
    } else if (SndOp snd = dyn_cast<SndOp>(apply.fun().getDefiningOp())) {
      emitRemark(expr.getLoc()) << "lower of Applied Snd";
      return lowerSnd(apply.getOperand(1), snd.getLoc(), rewriter);
    } else if (SplitOp splitOp =
                   dyn_cast<SplitOp>(apply.fun().getDefiningOp())) {
      emitRemark(expr.getLoc()) << "lower of Applied split";
      return lowerSplit(splitOp.nAttr(), splitOp.mAttr(), splitOp.tAttr(),
                        apply.getType(), apply.getOperand(1), splitOp.getLoc(),
                        rewriter);
    } else if (JoinOp joinOp = dyn_cast<JoinOp>(apply.fun().getDefiningOp())) {
      emitRemark(expr.getLoc()) << "ConT of Applied join";
      auto arrayCont =
          lower(apply.getOperand(1), rewriter.getInsertionPoint(), rewriter);
      auto joinInterm = rewriter.create<JoinIntermediateOp>(
          joinOp.getLoc(), apply.getType(), arrayCont, joinOp.nAttr(),
          joinOp.mAttr(), joinOp.tAttr());
      return joinInterm;
    } else if (TransposeOp transposeOp =
                   dyn_cast<TransposeOp>(apply.fun().getDefiningOp())) {
      emitRemark(expr.getLoc()) << "lower of Applied transpose";
      return lowerTranspose(
          transposeOp.nAttr(), transposeOp.mAttr(), transposeOp.tAttr(),
          apply.getType(), apply.getOperand(1), transposeOp.getLoc(), rewriter);
    } else if (SlideOp slideOp =
                   dyn_cast<SlideOp>(apply.fun().getDefiningOp())) {
      emitRemark(expr.getLoc()) << "lower of Applied Slide";
      return lowerSlide(slideOp.nAttr(), slideOp.szAttr(), slideOp.spAttr(),
                        slideOp.tAttr(), apply.getType(), apply.getOperand(1),
                        slideOp.getLoc(), rewriter);
    } else if (PadOp padOp = dyn_cast<PadOp>(apply.fun().getDefiningOp())) {
      emitRemark(expr.getLoc()) << "lower of Applied Pad";
      return lowerPad(padOp.nAttr(), padOp.lAttr(), padOp.rAttr(),
                      padOp.tAttr(), apply.getType(), apply.getOperand(1),
                      padOp.getLoc(), rewriter);
    } else if (MapSeqOp mapSeqOp =
                   dyn_cast<MapSeqOp>(apply.fun().getDefiningOp())) {
      emitRemark(expr.getLoc()) << "lower of Applied MapSeq";
      return lowerMapSeq(mapSeqOp.nAttr(), apply.getResult(), mapSeqOp.getLoc(),
                         rewriter);
    } else if (MapParOp mapParOp =
                   dyn_cast<MapParOp>(apply.fun().getDefiningOp())) {
      emitRemark(expr.getLoc())
          << "lower of Applied MapPar, currently not generating parallel code!";
      return lowerMapSeq(mapParOp.nAttr(), apply.getResult(), mapParOp.getLoc(),
                         rewriter);
    } else if (MapOp mapOp = dyn_cast<MapOp>(apply.fun().getDefiningOp())) {
      emitRemark(expr.getLoc()) << "lower of Applied Map";
      return lowerMap(mapOp.nAttr(), mapOp.sAttr(), mapOp.tAttr(),
                      dyn_cast<LambdaOp>(apply.getOperand(1).getDefiningOp()),
                      apply.getOperand(2), apply.getType(), mapOp.getLoc(),
                      rewriter);
    }
    emitError(apply.getLoc()) << "Cannot perform lowering for this apply!";
    return expr;
  } else if (EmbedOp embedOp = dyn_cast<EmbedOp>(expr.getDefiningOp())) {
    emitRemark(expr.getLoc()) << "lower of Embed";

    // Translating all operands
    for (int i = 0; i < embedOp.getOperands().size(); i++) {
      auto operand = embedOp.getOperand(i);
      auto operandCont = lower(operand, rewriter.getInsertionPoint(), rewriter);
      embedOp.setOperand(i, operandCont);
    }

    // This the embedded operations will be inlined later
    return embedOp.getResult();
  } else if (InOp inOp = dyn_cast<InOp>(expr.getDefiningOp())) {
    emitRemark(expr.getLoc()) << "lower of In";

    // we dont return the lowered of this to stay in our type system. This will
    // be resolved in the codeGen stage of the translation.
    return expr;
  }

  emitRemark(expr.getLoc())
      << "cannot perform lowering of "
      << expr.getDefiningOp()->getName().getStringRef().str()
      << " leaving Value as is.";

  rewriter.restoreInsertionPoint(oldInsertPoint);
  return expr;
}

//===----------------------------------------------------------------------===//
// LowerAndStore{Operation}
//===----------------------------------------------------------------------===//

void lowerAndStoreReduceSeq(NatAttr n, DataTypeAttr s, DataTypeAttr t,
                            Value reductionFun, Value initializer, Value array,
                            Value out, Location loc, PatternRewriter &rewriter,
                            StringRef loweringTarget) {
  using namespace mlir::edsc::op;
  using namespace mlir::edsc::intrinsics;
  OpBuilder builder(rewriter.getContext());
  ScopedContext scope(builder, loc);

  // Add Continuation for array.
  auto loweredArray = lower(array, rewriter.getInsertionPoint(), rewriter);
  auto cst_zero = std_constant_index(0);
  auto contInit = lower(initializer, rewriter.getInsertionPoint(), rewriter);

  // Introduce a temporary to accumulate into, or accumulate direcly in the
  // output
  bool defineNewAccumulator = false;

  Value accum;
  if (defineNewAccumulator) {

    EmbedOp embedOp = rewriter.create<EmbedOp>(
        loc,
        ScalarType::get(rewriter.getContext(),
                        FloatType::getF32(rewriter.getContext())),
        ValueRange());
    rewriter.setInsertionPointToStart(&embedOp.region().front());

    AllocOp alloc = rewriter.create<AllocOp>(
        loc, MemRefType::get(ArrayRef<int64_t>{0},
                             FloatType::getF32(rewriter.getContext())));

    rewriter.create<linalg::FillOp>(initializer.getLoc(), alloc.getResult(),
                                    contInit);
    rewriter.create<rise::ReturnOp>(initializer.getLoc(), alloc.getResult());

    rewriter.setInsertionPointAfter(embedOp);
    accum = embedOp.getResult();
  } else {
    accum = out;
  }
  auto initAccum = rewriter.create<AssignOp>(loc, contInit, accum);

  Value loopInductionVar;
  Block *forLoopBody;

  if (loweringTarget == "scf") {
    auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
    auto upperBound =
        rewriter.create<ConstantIndexOp>(loc, n.getValue().getIntValue());
    auto step = rewriter.create<ConstantIndexOp>(loc, 1);

    auto forLoop =
        rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step);
    loopInductionVar = forLoop.getInductionVar();
    forLoopBody = forLoop.getBody();
  } else if (loweringTarget == "affine") {
    auto forLoop =
        rewriter.create<AffineForOp>(loc, 0, n.getValue().getIntValue(), 1);
    loopInductionVar = forLoop.getInductionVar();
    forLoopBody = forLoop.getBody();
  }
  rewriter.setInsertionPointToStart(forLoopBody);

  LambdaOp reductionLambda = dyn_cast<LambdaOp>(reductionFun.getDefiningOp());

  IdxOp xi = rewriter.create<IdxOp>(
      loc, loweredArray.getType().cast<ArrayType>().getElementType(),
      loweredArray, loopInductionVar);

  LambdaOp lambdaCopy = cast<LambdaOp>(rewriter.clone(*reductionLambda));
  auto fxi = rewriter.create<ApplyOp>(loc, lambdaCopy.getType(),
                                      lambdaCopy.getResult(),
                                      ValueRange{accum, xi.getResult()});

  // Lower and store the application of accum and xi to the reduction lambda
  lowerAndStore(fxi.getResult(), accum, rewriter);

  // Copy of Lambda and corresponding Apply not needed anymore.
  fxi.getResult().dropAllUses();
  rewriter.eraseOp(fxi);
  lambdaCopy.getResult().dropAllUses();
  rewriter.eraseOp(lambdaCopy);

  // copy accumulator to output
  if (defineNewAccumulator) {
    rewriter.setInsertionPointAfter(forLoopBody->getParentOp());
    rewriter.create<AssignOp>(loc, accum, out);
  }
} // namespace

void lowerAndStoreMapSeq(NatAttr n, DataTypeAttr s, DataTypeAttr t, Value f,
                         Value array, Value out, Location loc,
                         PatternRewriter &rewriter, StringRef loweringTarget) {

  auto contArray = lower(array, rewriter.getInsertionPoint(), rewriter);

  // zero constant for indexing
  auto cst_zero = rewriter.create<ConstantIndexOp>(loc, 0);

  Value loopInductionVar;
  Block *forLoopBody;

  if (loweringTarget == "scf") {
    auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
    auto upperBound =
        rewriter.create<ConstantIndexOp>(loc, n.getValue().getIntValue());
    auto step = rewriter.create<ConstantIndexOp>(loc, 1);

    auto forLoop =
        rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step);
    loopInductionVar = forLoop.getInductionVar();
    forLoopBody = forLoop.getBody();
  } else if (loweringTarget == "affine") {
    auto forLoop =
        rewriter.create<AffineForOp>(loc, 0, n.getValue().getIntValue(), 1);
    loopInductionVar = forLoop.getInductionVar();
    forLoopBody = forLoop.getBody();
  }

  rewriter.setInsertionPointToStart(forLoopBody);

  LambdaOp fLambda = dyn_cast<LambdaOp>(f.getDefiningOp());

  IdxOp xi;

  if (contArray.getType().isa<ArrayType>()) {
    xi = rewriter.create<IdxOp>(
        loc, contArray.getType().cast<ArrayType>().getElementType(), contArray,
        loopInductionVar);
  }

  // operate on a copy of the lambda to avoid generating dependencies.
  LambdaOp lambdaCopy = cast<LambdaOp>(rewriter.clone(*fLambda));
  auto fxi = rewriter.create<ApplyOp>(loc, lambdaCopy.getType(),
                                      lambdaCopy.getResult(), xi.getResult());

  auto outi = rewriter.create<IdxOp>(
      loc, out.getType().dyn_cast<ArrayType>().getElementType(), out,
      loopInductionVar);

  // Lower and store the application of the outi to the mapped lambda
  lowerAndStore(fxi.getResult(), outi.getResult(), rewriter);

  // Copy of Lambda and corresponding Apply not needed anymore.
  fxi.getResult().dropAllUses();
  rewriter.eraseOp(fxi);
  lambdaCopy.getResult().dropAllUses();
  rewriter.eraseOp(lambdaCopy);

  rewriter.setInsertionPointAfter(forLoopBody->getParentOp());
}

void lowerAndStoreFst(Value tuple, Value out, Location loc,
                      PatternRewriter &rewriter) {
  auto contTuple = lower(tuple, rewriter.getInsertionPoint(), rewriter);
  auto fstIntermOp = rewriter.create<FstIntermediateOp>(
      loc, FloatType::getF32(rewriter.getContext()), contTuple);
  rewriter.create<AssignOp>(loc, fstIntermOp.getResult(), out);
}

void lowerAndStoreSnd(Value tuple, Value out, Location loc,
                      PatternRewriter &rewriter) {
  auto contTuple = lower(tuple, rewriter.getInsertionPoint(), rewriter);
  auto sndIntermOp = rewriter.create<SndIntermediateOp>(
      loc, FloatType::getF32(rewriter.getContext()), contTuple);
  rewriter.create<AssignOp>(loc, sndIntermOp.getResult(), out);
}

void lowerAndStoreSplit(NatAttr n, NatAttr m, DataTypeAttr t, Value array,
                        Value out, Location loc, PatternRewriter &rewriter) {
  ArrayType splitAccType = ArrayType::get(
      rewriter.getContext(),
      Nat::get(rewriter.getContext(),
               n.getValue().getIntValue() * m.getValue().getIntValue()),
      t.getValue());
  auto splitAccInterm =
      rewriter.create<SplitAccIntermediateOp>(loc, splitAccType, out, n, m, t);
  lowerAndStore(array, splitAccInterm.getResult(), rewriter);
}

void lowerAndStoreJoin(NatAttr n, NatAttr m, DataTypeAttr t, Value array,
                       Value out, Location loc, PatternRewriter &rewriter) {
  ArrayType joinAccType = ArrayType::get(
      rewriter.getContext(), n.getValue(),
      ArrayType::get(rewriter.getContext(), m.getValue(), t.getValue()));

  auto joinAccInterm =
      rewriter.create<JoinAccIntermediateOp>(loc, joinAccType, out, n, m, t);
  lowerAndStore(array, joinAccInterm.getResult(), rewriter);
}

//===----------------------------------------------------------------------===//
// Lower{Operation}
//===----------------------------------------------------------------------===//

mlir::Value lowerLiteral(Value literalValue, Location loc,
                         PatternRewriter &rewriter) {

  std::string literalString = dyn_cast<LiteralOp>(literalValue.getDefiningOp())
                                  .literalAttr()
                                  .getValue();
  DataType literalType =
      dyn_cast<LiteralOp>(literalValue.getDefiningOp()).literalAttr().getType();

  if (literalType.isa<ScalarType>()) {
    EmbedOp embedOp = rewriter.create<EmbedOp>(
        loc,
        ScalarType::get(rewriter.getContext(),
                        FloatType::getF32(rewriter.getContext())),
        ValueRange());
    rewriter.setInsertionPointToStart(&embedOp.region().front());

    auto fillOp = rewriter.create<ConstantFloatOp>(
        loc, llvm::APFloat(std::stof(literalString)),
        FloatType::getF32(rewriter.getContext()));
    rewriter.create<rise::ReturnOp>(loc, fillOp.getResult());

    return embedOp.getResult();
  } else if (ArrayType arrayType = literalType.dyn_cast<ArrayType>()) {
    SmallVector<int64_t, 4> shape = {};

    shape.push_back(arrayType.getSize().getIntValue());
    while (arrayType.getElementType().isa<ArrayType>()) {
      arrayType = arrayType.getElementType().dyn_cast<ArrayType>();
      shape.push_back(arrayType.getSize().getIntValue());
    }

    ScalarType elementType = arrayType.getElementType().dyn_cast<ScalarType>();
    assert(
        elementType &&
        "Element type of inner array of Literal has to be of Type ScalarType!");
    Type memrefElementType = elementType.getWrappedType();

    auto array = rewriter.create<AllocOp>(
        loc, MemRefType::get(shape, memrefElementType));

    // For now just fill the array with one value
    StringRef litStr = literalString;
    litStr = litStr.substr(0, litStr.find_first_of(','));
    litStr = litStr.trim('[');
    litStr = litStr.trim(']');
    float fillValue = std::stof(litStr.str() + ".0f");

    auto filler = rewriter.create<ConstantFloatOp>(
        loc, llvm::APFloat(fillValue),
        FloatType::getF32(rewriter.getContext()));

    rewriter.create<linalg::FillOp>(loc, array.getResult(), filler.getResult());

    return array.getResult();
  } else {
    emitError(loc) << "Lowering literals of this type not supported right now!";
    return nullptr;
  }
}

mlir::Value lowerMapSeq(NatAttr n, Value val, Location loc,
                        PatternRewriter &rewriter) {
  // introduce tmp Array of length n:
  EmbedOp embedOp = rewriter.create<EmbedOp>(
      loc,
      ArrayType::get(rewriter.getContext(), n.getValue(),
                     ScalarType::get(rewriter.getContext(),
                                     FloatType::getF32(rewriter.getContext()))),
      ValueRange());

  rewriter.setInsertionPointToStart(&embedOp.region().front());
  auto tmpArray = rewriter.create<AllocOp>(
      loc, MemRefType::get(ArrayRef<int64_t>{n.getValue().getIntValue()},
                           FloatType::getF32(rewriter.getContext())));
  rewriter.create<rise::ReturnOp>(tmpArray.getLoc(), tmpArray.getResult());

  rewriter.setInsertionPointAfter(embedOp);
  lowerAndStore(val, embedOp.getResult(), rewriter);

  return embedOp.getResult();
}

mlir::Value lowerMap(NatAttr n, DataTypeAttr s, DataTypeAttr t, LambdaOp mapFun,
                     Value array, Type type, Location loc,
                     PatternRewriter &rewriter) {

  auto contArray = lower(array, rewriter.getInsertionPoint(), rewriter);

  // The idx which is needed in the lambda can only be generated in the
  // second part of the codegen, so we use a placeholder for it here.
  auto placeholder = rewriter.create<PlaceholderOp>(
      loc, mapFun.region().front().getArgument(0).getType());
  Substitute(mapFun, {placeholder});

  rise::ReturnOp returnOp =
      dyn_cast<rise::ReturnOp>(mapFun.region().front().getTerminator());
  auto loweredMapFun =
      lower(returnOp.getOperand(0), rewriter.getInsertionPoint(), rewriter);
  return rewriter
      .create<MapReadIntermediateOp>(loc, type, n, s, t, loweredMapFun,
                                     placeholder, contArray)
      .getResult();
}

mlir::Value lowerZip(Value lhs, Value rhs, Location loc,
                     PatternRewriter &rewriter) {
  auto contLhs = lower(lhs, rewriter.getInsertionPoint(), rewriter);
  auto contRhs = lower(rhs, rewriter.getInsertionPoint(), rewriter);

  ArrayType lhType = lhs.getType().dyn_cast<ArrayType>();
  ArrayType rhType = rhs.getType().dyn_cast<ArrayType>();
  assert(lhType && rhType && "Inputs to zip have to be Arrays!");

  ArrayType zipType = ArrayType::get(rewriter.getContext(), lhType.getSize(),
                                     rise::Tuple::get(rewriter.getContext(),
                                                      lhType.getElementType(),
                                                      rhType.getElementType()));
  return rewriter.create<ZipIntermediateOp>(loc, zipType, contLhs, contRhs)
      .getResult();
}

mlir::Value lowerFst(Value tuple, Location loc, PatternRewriter &rewriter) {
  auto tupleCont = lower(tuple, rewriter.getInsertionPoint(), rewriter);
  return rewriter
      .create<FstIntermediateOp>(loc, FloatType::getF32(rewriter.getContext()),
                                 tupleCont)
      .getResult();
}

mlir::Value lowerSnd(Value tuple, Location loc, PatternRewriter &rewriter) {
  auto tupleCont = lower(tuple, rewriter.getInsertionPoint(), rewriter);
  return rewriter
      .create<SndIntermediateOp>(loc, FloatType::getF32(rewriter.getContext()),
                                 tupleCont)
      .getResult();
}

mlir::Value lowerSplit(NatAttr n, NatAttr m, DataTypeAttr t, Type type,
                       Value array, Location loc, PatternRewriter &rewriter) {
  auto arrayCont = lower(array, rewriter.getInsertionPoint(), rewriter);
  return rewriter.create<SplitIntermediateOp>(loc, type, arrayCont, n, m, t)
      .getResult();
}

mlir::Value lowerJoin(NatAttr n, NatAttr m, DataTypeAttr t, Type type,
                      Value array, Location loc, PatternRewriter &rewriter) {
  auto arrayCont = lower(array, rewriter.getInsertionPoint(), rewriter);
  auto joinInterm =
      rewriter.create<JoinIntermediateOp>(loc, type, arrayCont, n, m, t)
          .getResult();
}

mlir::Value lowerTranspose(NatAttr n, NatAttr m, DataTypeAttr t, Type type,
                           Value array, Location loc,
                           PatternRewriter &rewriter) {
  auto arrayCont = lower(array, rewriter.getInsertionPoint(), rewriter);
  return rewriter.create<TransposeIntermediateOp>(loc, type, arrayCont, n, m, t)
      .getResult();
}

mlir::Value lowerSlide(NatAttr n, NatAttr sz, NatAttr sp, DataTypeAttr t,
                       Type type, Value array, Location loc,
                       PatternRewriter &rewriter) {
  auto arrayCont = lower(array, rewriter.getInsertionPoint(), rewriter);
  return rewriter
      .create<SlideIntermediateOp>(loc, type, arrayCont, n, sz, sp, t)
      .getResult();
}

mlir::Value lowerPad(NatAttr n, NatAttr l, NatAttr r, DataTypeAttr t, Type type,
                     Value array, Location loc, PatternRewriter &rewriter) {
  auto arrayCont = lower(array, rewriter.getInsertionPoint(), rewriter);
  return rewriter.create<PadIntermediateOp>(loc, type, arrayCont, n, l, r, t)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Second part of the lowering process
//===----------------------------------------------------------------------===//

void lowerAssign(AssignOp assignOp, PatternRewriter &rewriter) {
  Location loc = assignOp.getLoc();
  emitRemark(loc) << "Codegen for Assign";

  if (assignOp.value().isa<OpResult>()) {
    rewriter.setInsertionPoint(assignOp.assignee().getDefiningOp());
  } else {
    rewriter.setInsertionPointToStart(
        &assignOp.value().getParentRegion()->front());
  }
  auto writeValue = resolveIndexing(assignOp.value(), {}, rewriter);
  if (!writeValue)
    emitError(loc) << "Assignment has no Value to write.";

  rewriter.setInsertionPointAfter(assignOp);
  auto leftPath =
      resolveStoreIndexing(assignOp.assignee(), writeValue, {}, rewriter);
}

SmallVector<OutputPathType, 10>
resolveStoreIndexing(Value storeLocation, Value val,
                     SmallVector<OutputPathType, 10> path,
                     PatternRewriter &rewriter) {
  if (!storeLocation.isa<OpResult>()) {
    emitRemark(val.getLoc()) << "CodegenStore for BlockArg";
    generateReadAccess(path, val, storeLocation, rewriter);
    return path;
  }

  if (IdxOp idx = dyn_cast<IdxOp>(storeLocation.getDefiningOp())) {
    emitRemark(val.getLoc()) << "CodegenStore for idx";

    path.push_back(idx.iv());

    return resolveStoreIndexing(idx.array(), val, path, rewriter);
  } else if (CastOp castOp = dyn_cast<CastOp>(storeLocation.getDefiningOp())) {
    emitRemark(val.getLoc()) << "CodegenStore for cast";
    return resolveStoreIndexing(castOp.getOperand(), val, path, rewriter);
  } else if (JoinAccIntermediateOp joinAccOp = dyn_cast<JoinAccIntermediateOp>(
                 storeLocation.getDefiningOp())) {
    emitRemark(val.getLoc()) << "CodegenStore for joinAcc";
    auto i = mpark::get<Value>(path.pop_back_val());
    auto j = mpark::get<Value>(path.pop_back_val());

    auto cstM = rewriter
                    .create<ConstantIndexOp>(joinAccOp.getLoc(),
                                             joinAccOp.m().getIntValue())
                    .getResult();
    auto i_times_m =
        rewriter.create<MulIOp>(joinAccOp.getLoc(), i, cstM).getResult();
    auto newIndex =
        rewriter.create<AddIOp>(joinAccOp.getLoc(), i_times_m, j).getResult();

    path.push_back(newIndex);
    return resolveStoreIndexing(joinAccOp.getOperand(), val, path, rewriter);
  } else if (SplitAccIntermediateOp splitAccOp =
                 dyn_cast<SplitAccIntermediateOp>(
                     storeLocation.getDefiningOp())) {
    emitRemark(val.getLoc()) << "CodegenStore for splitAcc";
    auto loc = splitAccOp.getLoc();
    auto lhs = mpark::get<Value>(path.pop_back_val());
    auto rhs =
        rewriter.create<ConstantIndexOp>(loc, splitAccOp.n().getIntValue())
            .getResult();

    // modulo op taken from AffineToStandard
    Value remainder = rewriter.create<SignedRemIOp>(loc, lhs, rhs);
    Value zeroCst = rewriter.create<ConstantIndexOp>(loc, 0);
    Value isRemainderNegative =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, remainder, zeroCst);
    Value correctedRemainder = rewriter.create<AddIOp>(loc, remainder, rhs);
    Value result = rewriter.create<SelectOp>(loc, isRemainderNegative,
                                             correctedRemainder, remainder);
    Value divResult = rewriter.create<UnsignedDivIOp>(loc, lhs, rhs);

    path.push_back(result);
    path.push_back(divResult);
    return resolveStoreIndexing(splitAccOp.getOperand(), val, path, rewriter);
  } else if (EmbedOp embedOp =
                 dyn_cast<EmbedOp>(storeLocation.getDefiningOp())) {
    emitRemark(val.getLoc()) << "CodegenStore for embed";
    assert(embedOp.getNumOperands() == 0 &&
           "codegenstore for embed with operands not handled yet.");

    auto oldInsertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(embedOp);

    // replace uses of embed value with returned value
    rise::ReturnOp embedReturn = dyn_cast<rise::ReturnOp>(
        embedOp.getRegion().front().getOperations().back());
    assert(embedReturn &&
           "Region of EmbedOp has to be terminated using rise.return!");
    embedOp.getResult().replaceAllUsesWith(embedReturn.getOperand(0));

    // inline all operations of the embedOp region except for return
    rewriter.getInsertionBlock()->getOperations().splice(
        rewriter.getInsertionPoint(),
        embedOp.getRegion().front().getOperations(),
        embedOp.getRegion().front().begin(), Block::iterator(embedReturn));

    rewriter.restoreInsertionPoint(oldInsertPoint);
    generateReadAccess(path, val, embedReturn.getOperand(0), rewriter);
    return path;
  }

  emitRemark(val.getLoc())
      << "CodegenStore for "
      << val.getDefiningOp()->getName().getStringRef().str();

  generateReadAccess(path, val, storeLocation, rewriter);
  return path;
}

Value resolveIndexing(Value val, SmallVector<OutputPathType, 10> path,
                      PatternRewriter &rewriter) {

  if (!val.isa<OpResult>()) {
    emitRemark(val.getLoc()) << "reached a blockArg in Codegen, reversing";
    return generateWriteAccess(path, val, rewriter);
  }
  if (EmbedOp embedOp = dyn_cast<EmbedOp>(val.getDefiningOp())) {
    emitRemark(embedOp.getLoc()) << "Codegen for Embed";

    auto oldInsertPoint = rewriter.saveInsertionPoint();

    rewriter.setInsertionPointAfter(embedOp);

    // Do codegen for all operands of embed first
    int i = 0;
    for (auto operand : (val.getDefiningOp()->getOperands())) {
      embedOp.setOperand(i, resolveIndexing(operand, path, rewriter));
      i++;
    }
    // replace blockArgs in the region with results of the codegen for the
    // operands
    for (int i = 0; i < embedOp.getOperands().size(); i++) {
      embedOp.region().front().getArgument(i).replaceAllUsesWith(
          embedOp.getOperand(i));
    }
    // replace uses of embed value with returned value
    rise::ReturnOp embedReturn = dyn_cast<rise::ReturnOp>(
        embedOp.getRegion().front().getOperations().back());
    embedOp.getParentOfType<FuncOp>().dump();
    assert(embedReturn &&
           "Region of EmbedOp has to be terminated using rise.return!");
    embedOp.getResult().replaceAllUsesWith(embedReturn.getOperand(0));

    // inline all operations of the embedOp region except for return
    rewriter.getInsertionBlock()->getOperations().splice(
        rewriter.getInsertionPoint(),
        embedOp.getRegion().front().getOperations(),
        embedOp.getRegion().front().begin(), Block::iterator(embedReturn));

    rewriter.restoreInsertionPoint(oldInsertPoint);

    return embedReturn.getOperand(0);

  } else if (IdxOp idx = dyn_cast<IdxOp>(val.getDefiningOp())) {
    // printPath(path, "idx");

    emitRemark(idx.getLoc()) << "Codegen for idx";

    Value iv = idx.iv();
    path.push_back(iv);
    return resolveIndexing(idx.array(), path, rewriter);
  } else if (AllocOp alloc = dyn_cast<AllocOp>(val.getDefiningOp())) {
    emitRemark(alloc.getLoc()) << "Codegen for alloc";

    // call to reverse here.
    return generateWriteAccess(path, alloc.getResult(), rewriter);
  } else if (MapReadIntermediateOp mapReadOp =
                 dyn_cast<MapReadIntermediateOp>(val.getDefiningOp())) {
    // printPath(path, "MapRead:");

    emitRemark(mapReadOp.getLoc()) << "Codegen for MapRead";

    auto i = mpark::get<Value>(path.pop_back_val());
    auto idx = rewriter.create<IdxOp>(
        mapReadOp.getLoc(),
        mapReadOp.array().getType().dyn_cast<ArrayType>().getElementType(),
        mapReadOp.array(), i);

    mapReadOp.placeholder().replaceAllUsesWith(idx.getResult());
    return resolveIndexing(mapReadOp.f(), path, rewriter);
  } else if (ZipIntermediateOp zipIntermOp =
                 dyn_cast<ZipIntermediateOp>(val.getDefiningOp())) {
    emitRemark(zipIntermOp.getLoc()) << "Codegen for zip";
    OutputPathType sndLastElem = path[path.size() - 2];
    int *fst = mpark::get_if<int>(&sndLastElem);

    // delete snd value on the path.
    auto tmp = path.pop_back_val();
    path.pop_back();
    path.push_back(tmp);

    if (*fst) {
      return resolveIndexing(zipIntermOp.lhs(), path, rewriter);
    } else {
      return resolveIndexing(zipIntermOp.rhs(), path, rewriter);
    }
  } else if (FstIntermediateOp fstIntermOp =
                 dyn_cast<FstIntermediateOp>(val.getDefiningOp())) {
    // printPath(path, "fst");
    emitRemark(fstIntermOp.getLoc()) << "Codegen for fst";

    path.push_back(true);
    return resolveIndexing(fstIntermOp.value(), path, rewriter);

  } else if (SndIntermediateOp sndIntermOp =
                 dyn_cast<SndIntermediateOp>(val.getDefiningOp())) {
    // printPath(path, "snd");
    emitRemark(sndIntermOp.getLoc()) << "Codegen for snd";

    path.push_back(false);
    return resolveIndexing(sndIntermOp.value(), path, rewriter);

  } else if (isa<LoadOp>(val.getDefiningOp()) ||
             isa<AffineLoadOp>(val.getDefiningOp())) {
    emitRemark(val.getLoc()) << "Codegen for Load";
    return val;
  } else if (SplitIntermediateOp splitIntermediateOp =
                 dyn_cast<SplitIntermediateOp>(val.getDefiningOp())) {
    emitRemark(val.getLoc()) << "Codegen for Split";

    auto i = mpark::get<Value>(path.pop_back_val());
    auto j = mpark::get<Value>(path.pop_back_val());

    auto cstN =
        rewriter
            .create<ConstantIndexOp>(splitIntermediateOp.getLoc(),
                                     splitIntermediateOp.n().getIntValue())
            .getResult();
    auto i_times_n =
        rewriter.create<MulIOp>(splitIntermediateOp.getLoc(), i, cstN)
            .getResult();
    auto newIndex =
        rewriter.create<AddIOp>(splitIntermediateOp.getLoc(), i_times_n, j)
            .getResult();
    path.push_back(newIndex);

    return resolveIndexing(splitIntermediateOp.value(), path, rewriter);
  } else if (JoinIntermediateOp joinIntermediateOp =
                 dyn_cast<JoinIntermediateOp>(val.getDefiningOp())) {
    emitRemark(val.getLoc()) << "Codegen for Join";

    auto loc = joinIntermediateOp.getLoc();
    auto lhs = mpark::get<Value>(path.pop_back_val());
    auto rhs =
        rewriter
            .create<ConstantIndexOp>(loc, joinIntermediateOp.n().getIntValue())
            .getResult();

    // modulo op taken from AffineToStandard
    Value remainder = rewriter.create<SignedRemIOp>(loc, lhs, rhs);
    Value zeroCst = rewriter.create<ConstantIndexOp>(loc, 0);
    Value isRemainderNegative =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, remainder, zeroCst);
    Value correctedRemainder = rewriter.create<AddIOp>(loc, remainder, rhs);
    Value result = rewriter.create<SelectOp>(loc, isRemainderNegative,
                                             correctedRemainder, remainder);
    Value divResult = rewriter.create<SignedDivIOp>(loc, lhs, rhs);

    path.push_back(result);
    path.push_back(divResult);

    return resolveIndexing(joinIntermediateOp.value(), path, rewriter);
  } else if (TransposeIntermediateOp transposeIntermediateOp =
                 dyn_cast<TransposeIntermediateOp>(val.getDefiningOp())) {
    // printPath(path, "Transpose:");
    emitRemark(val.getLoc()) << "Codegen for Transpose";
    auto n = path.pop_back_val();
    auto m = path.pop_back_val();

    path.push_back(n);
    path.push_back(m);

    return resolveIndexing(transposeIntermediateOp.getOperand(), path,
                           rewriter);
  } else if (SlideIntermediateOp slideIntermediateOp =
                 dyn_cast<SlideIntermediateOp>(val.getDefiningOp())) {
    // printPath(path, "Slide:");

    emitRemark(val.getLoc()) << "Codegen for Slide";

    Value i = mpark::get<Value>(path.pop_back_val());
    Value j = mpark::get<Value>(path.pop_back_val());

    if (!j) {
      emitError(slideIntermediateOp.getLoc())
          << "Cannot do codegen for slide, path structure not correct!";
    }

    Value s2 =
        rewriter
            .create<ConstantIndexOp>(slideIntermediateOp.getLoc(),
                                     slideIntermediateOp.sp().getIntValue())
            .getResult();

    Value i_times_s2 =
        rewriter.create<MulIOp>(slideIntermediateOp.getLoc(), i, s2)
            .getResult();

    Value newIndex =
        rewriter.create<AddIOp>(slideIntermediateOp.getLoc(), i_times_s2, j)
            .getResult();
    path.push_back(newIndex);

    return resolveIndexing(slideIntermediateOp.value(), path, rewriter);
  } else if (PadIntermediateOp padIntermediateOp =
                 dyn_cast<PadIntermediateOp>(val.getDefiningOp())) {
    emitRemark(val.getLoc()) << "Codegen for Pad";
    Location loc = padIntermediateOp.getLoc();

    Value i = mpark::get<Value>(path.pop_back_val());

    // I will do padclamp first
    //      Value padVal = resolveIndexing(padIntermediateOp.padvalue(),
    //      path,rewriter);

    Value l =
        rewriter
            .create<ConstantIndexOp>(loc, padIntermediateOp.l().getIntValue())
            .getResult();
    Value r =
        rewriter
            .create<ConstantIndexOp>(loc, padIntermediateOp.r().getIntValue())
            .getResult();
    Value n =
        rewriter
            .create<ConstantIndexOp>(loc, padIntermediateOp.n().getIntValue())
            .getResult();

    Value cst0 = rewriter.create<ConstantIndexOp>(loc, 0).getResult();
    Value cst1 = rewriter.create<ConstantIndexOp>(loc, 1).getResult();
    Value n_minus_1 = rewriter.create<SubIOp>(loc, n, cst1).getResult();
    Value l_plus_n = rewriter.create<AddIOp>(loc, l, n).getResult();
    Value index = rewriter.create<SubIOp>(loc, i, l);

    // Conditions whether index is in bounds.
    // We can just base the conditions on i, not on i+l as done in shine_rise.
    Value isIndexSmallerLb =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, i, l);
    Value isIndexSmallerRb =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, i, l_plus_n);

    // (i < l) ? 0 : (i < l+n) ? index : n-1
    Value thenelseVal =
        rewriter.create<SelectOp>(loc, isIndexSmallerRb, index, n_minus_1);
    Value ifthenVal =
        rewriter.create<SelectOp>(loc, isIndexSmallerLb, cst0, thenelseVal);

    path.push_back(ifthenVal);

    return resolveIndexing(padIntermediateOp.array(), path, rewriter);
  } else if (InOp inOp = dyn_cast<InOp>(val.getDefiningOp())) {
    emitRemark(val.getLoc())
        << "Codegen for In, generating read operation for operand";
    return generateWriteAccess(path, inOp.input(), rewriter);
  } else if (CastOp castOp = dyn_cast<CastOp>(val.getDefiningOp())) {
    emitRemark(val.getLoc()) << "Codegen for Cast, reversing";
    return generateWriteAccess(path, castOp.getOperand(), rewriter);
  }

  emitRemark(val.getLoc())
      << "I don't know how to do codegen for: "
      << val.getDefiningOp()->getName().getStringRef().str()
      << " this is prob. an operation from another dialect. We walk "
         "recursively through the operands until we hit something we can "
         "do codegen for.";

  int i = 0;
  for (auto operand : (val.getDefiningOp()->getOperands())) {
    val.getDefiningOp()->setOperand(i,
                                    resolveIndexing(operand, path, rewriter));
    i++;
  }
  return val;
}

Value generateWriteAccess(SmallVector<OutputPathType, 10> path, Value accessVal,
                          PatternRewriter &rewriter) {
  SmallVector<Value, 10> indexValues = {};
  //  for (OutputPathType element : path) {
  for (auto element = path.rbegin(); element != path.rend(); ++element) {
    auto val = mpark::get_if<Value>(&*element);
    assert(val && "path is ill structured!");
    indexValues.push_back(*val);
  }

  if (isa<AffineForOp>(rewriter.getBlock()->getParent()->getParentOp())) {
    return rewriter
        .create<AffineLoadOp>(accessVal.getLoc(), accessVal, indexValues)
        .getResult();
  } else {
    return rewriter.create<LoadOp>(accessVal.getLoc(), accessVal, indexValues)
        .getResult();
  }
}

void generateReadAccess(SmallVector<OutputPathType, 10> path, Value storeVal,
                        Value storeLoc, PatternRewriter &rewriter) {
  SmallVector<Value, 10> indexValues = {};
  for (auto element = path.rbegin(); element != path.rend(); ++element) {
    auto val = mpark::get_if<Value>(&*element);
    assert(val && "path is ill structured!");
    indexValues.push_back(*val);
  }
  int rank = storeLoc.getType().dyn_cast<MemRefType>().getRank();
  if (indexValues.size() != rank) {
    indexValues.erase(indexValues.begin());
  }
  if (isa<AffineForOp>(rewriter.getBlock()->getParent()->getParentOp())) {
    rewriter.create<AffineStoreOp>(storeLoc.getLoc(), storeVal, storeLoc,
                                   llvm::makeArrayRef(indexValues));
    return;
  } else {
    rewriter.create<StoreOp>(storeLoc.getLoc(), storeVal, storeLoc,
                             llvm::makeArrayRef(indexValues));

    return;
  }
}

void printPath(SmallVector<OutputPathType, 10> path, StringRef additionalInfo) {
  struct {
    void operator()(Value val) {
      if (val.isa<OpResult>()) {
        std::cout << "val: "
                  << val.getDefiningOp()->getName().getStringRef().str();
      } else {
        std::cout << "blockArg";
      }
    }
    void operator()(int i) { i ? std::cout << "fst" : std::cout << "snd"; }
    //    void operator()(std::string const &) { std::cout << "string!"; }
    //    void operator()(bool b) { std::cout << "bool: " << b <; }
  } visitor;
  std::cout << "path: " << additionalInfo.str() << " {";
  for (int i = 0; i < path.size(); i++) {
    mpark::visit(visitor, path[i]);
    if (i < path.size() - 1)
      std::cout << ", ";
  }
  std::cout << "}\n" << std::flush;
}

void printUses(Value val) {
  std::cout << val.getDefiningOp()->getName().getStringRef().str()
            << " has uses: \n"
            << std::flush;

  auto uses = val.getUses().begin();
  while (true) {
    if (uses != val.getUses().end()) {
      std::cout << "    " << uses.getUser()->getName().getStringRef().str()
                << "\n"
                << std::flush;
      uses++;
    } else {
      break;
    }
  }
}

void Substitute(LambdaOp lambda, llvm::SmallVector<Value, 10> args) {
  if (lambda.region().front().getArguments().size() < args.size()) {
    emitError(lambda.getLoc())
        << "Too many arguments given for Lambda substitution";
  }
  for (int i = 0; i < args.size(); i++) {
    lambda.region().front().getArgument(i).replaceAllUsesWith(args[i]);
  }
  return;
}

} // namespace

/// gather all patterns
void mlir::populateRiseToImpConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<RiseToImperativePattern>(ctx);
}
//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// The pass:
void ConvertRiseToImperativePass::runOnFunction() {
  auto module = getOperation();
  OwningRewritePatternList patterns;

  populateRiseToImpConversionPatterns(patterns, &getContext());

  ConversionTarget target(getContext());

  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<linalg::LinalgDialect>();
  target.addLegalDialect<rise::RiseDialect>(); // for debugging purposes

  target.addDynamicallyLegalOp<FuncOp>([](FuncOp funcOp) {
    bool riseInside = false;
    if (funcOp.isExternal())
      return true;
    funcOp.walk([&](Operation *op) {
      if (op->getDialect()->getNamespace().equals(
              rise::RiseDialect::getDialectNamespace()))
        riseInside = true;
    });
    return !riseInside;
  });

  bool erased;
  applyOpPatternsAndFold(module, patterns, &erased);

  return;
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::rise::createConvertRiseToImperativePass() {
  return std::make_unique<ConvertRiseToImperativePass>();
}
