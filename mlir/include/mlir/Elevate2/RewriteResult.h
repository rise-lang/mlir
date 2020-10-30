//
// Created by martin on 30/10/2020.
//

#ifndef LLVM_REWRITERESULT_H
#define LLVM_REWRITERESULT_H

#include "mlir/IR/OpDefinition.h"
#include <memory>
#include <stdexcept>
#include <variant>

namespace mlir {
namespace elevate {

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

template <class... Ts>
struct cases : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
cases(Ts...) -> cases<Ts...>;

template <class Variant, class Visitor>
decltype(auto) match(Variant &&var, Visitor &&vis) {
  return std::visit(std::forward<Visitor>(vis), std::forward<Variant>(var));
}

template <class Variant1, class Variant2, class Visitor>
decltype(auto) match(Variant1 &&var1, Variant2 &&var2, Visitor &&vis) {
  return std::visit(std::forward<Visitor>(vis), std::forward<Variant1>(var1),
                    std::forward<Variant2>(var2));
}

struct Success;
struct Failure;

using RewriteResult = std::variant<Success, Failure>; // will be LogicalResult

struct Success {
  Operation *op;
};

auto success(Operation *op) -> Success;

struct Failure {};

auto failure() -> Failure;

}
}

#endif // LLVM_REWRITERESULT_H
