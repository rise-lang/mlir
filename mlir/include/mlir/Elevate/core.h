//
// Created by martin on 9/25/20.
//

#ifndef LLVM_ELEVATE_CORE_H
#define LLVM_ELEVATE_CORE_H

#include <iostream>
#include <memory>
#include <variant>
#include <stdexcept>

struct Expr {};


template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template<class... Ts> struct cases : Ts... { using Ts::operator()...; };
template<class... Ts> cases(Ts...) -> cases<Ts...>;

template <class Variant, class Visitor>
auto match(Variant&& var, Visitor&& vis) {
  return std::visit(std::forward<Visitor>(vis), std::forward<Variant>(var));
}

template <class Variant1, class Variant2, class Visitor>
auto match(Variant1&& var1, Variant2&& var2, Visitor&& vis) {
  return std::visit(std::forward<Visitor>(vis),
                    std::forward<Variant1>(var1), std::forward<Variant2>(var2));
}


struct Success;
struct Failure;

using RewriteResult = std::variant<Success, Failure>;

struct Success {
  Expr expr;
};

auto success(const Expr& expr) -> Success {
  return Success{expr};
}

struct Failure {};

auto failure() -> Failure {
  return Failure{};
}

struct Strategy {
  virtual RewriteResult operator()(const Expr& expr) const = 0;
};

auto getExpr(RewriteResult rr) -> Expr {
  return match(rr, cases {
      [](const Success& s) {
        return s.expr;
      },
      [](const Failure& f) -> Expr {
        std::cout << "elevate logic error!\n";
//        throw std::logic_error("");
        return {}; // what Expr do we want to return here?
      }
  });
}

auto flatMapSuccess(RewriteResult rr, const Strategy& s) -> RewriteResult {
  return match(rr, cases {
      [&](const Success& ss) -> RewriteResult {
        return s(ss.expr);
      },
      [&](const Failure& f) -> RewriteResult {
        return f;
      }
  });
}

template <typename F>
auto flatMapFailure(RewriteResult rr, const F& f) -> RewriteResult {
  return match(rr, cases {
      [&](const Success& ss) -> RewriteResult {
        return ss;
      },
      [&](const Failure&) -> RewriteResult {
        return f();
      }
  });
}

struct IdStrategy : Strategy {
  RewriteResult operator()(const Expr& expr) const override {
    return success(expr);
  };
};

auto id = IdStrategy();

struct FailStrategy : Strategy {
  RewriteResult operator()(const Expr& expr) const override {
    return failure();
  };
};

auto fail = FailStrategy();

struct SeqStrategy : Strategy {
  const Strategy& fs;
  const Strategy& ss;

  SeqStrategy(const Strategy& fs, const Strategy& ss) : fs{fs}, ss{ss} {};

  RewriteResult operator()(const Expr& expr) const override {
    return flatMapSuccess(fs(expr), ss);
  };
};

auto seq = [](const auto& fs) { return [&](const auto& ss) { return SeqStrategy(fs, ss); }; };

struct LeftChoiceStrategy : Strategy {
  const Strategy& fs;
  const Strategy& ss;

  LeftChoiceStrategy(const Strategy& fs, const Strategy& ss) : fs{fs}, ss{ss} {};

  RewriteResult operator()(const Expr& expr) const override {
    return flatMapFailure(fs(expr), [&] { return ss(expr); });
  };
};

auto leftChoice = [](const auto& fs) { return [&](const auto& ss) { return LeftChoiceStrategy(fs, ss); }; };

struct TryStrategy : Strategy {
  const Strategy& s;

  TryStrategy(const Strategy& s): s{s} {};

  RewriteResult operator()(const Expr& expr) const override {
    return leftChoice(s)(id)(expr);
  };
};

auto try_ = [](const auto& s) { return TryStrategy(s); };

struct RepeatStrategy : Strategy {
  const Strategy& s;

  RepeatStrategy(const Strategy& s): s{s} {};

  RewriteResult operator()(const Expr& expr) const override {
    return try_(seq(s)(RepeatStrategy(s)))(expr);
  };
};

auto repeat = [](const auto& s) { return RepeatStrategy(s); };


#endif // LLVM_ELEVATE_CORE_H
