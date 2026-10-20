// @HEADER
// *****************************************************************************
//        Phalanx: A Partial Differential Equation Field Evaluation
//       Kernel for Flexible Management of Complex Dependency Chains
//
// Copyright 2008 NTESS and the Phalanx contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teuchos_UnitTestHarness.hpp"
#include "Phalanx_TemplateContainer.hpp"
#include "Phalanx_TemplateManager.hpp"
#include "Sacado_mpl_placeholders.hpp"
#include <string>
#include <vector>

namespace {

  struct Residual {};
  struct Jacobian {};
  struct Tangent {};
  using EvalTypes = PHX::mpl::type_list<Residual,Jacobian,Tangent>;

  template <typename T> std::string name();
  template <> std::string name<Residual>() { return "R"; }
  template <> std::string name<Jacobian>() { return "J"; }
  template <> std::string name<Tangent>() { return "T"; }

  struct Base {
    virtual ~Base() = default;
    virtual std::string describe() const = 0;
  };

  template <typename EvalT>
  struct Object : Base {
    virtual int order() const = 0;
  };

  template <typename EvalT>
  struct Concrete : Object<EvalT> {
    std::string label;
    int order_;
    Concrete(const std::string& l, int o) : label(l), order_(o) {}
    std::string describe() const override { return label + "<" + name<EvalT>() + ">"; }
    int order() const override { return order_; }
  };

  // Derives from Base but not from Object<EvalT>.
  struct NotAnObject : Base {
    std::string describe() const override { return "wrong"; }
  };

  // A builder in the PHX::TemplateManager style.
  struct LegacyBuilder {
    int order;
    template <typename EvalT>
    Teuchos::RCP<Base> build() const { return Teuchos::rcp(new Concrete<EvalT>("legacy",order)); }
  };

  // A PHX::TemplateManager-style builder whose build<T>() returns a
  // derived type, so no runtime type check is needed.
  struct TypedLegacyBuilder {
    template <typename EvalT>
    Teuchos::RCP<Concrete<EvalT>> build() const { return Teuchos::rcp(new Concrete<EvalT>("typed",5)); }
  };

  struct WrongTypeBuilder {
    template <typename EvalT>
    Teuchos::RCP<Base> build() const { return Teuchos::rcp(new NotAnObject); }
  };

  using Container = PHX::TemplateContainer<EvalTypes,Object,Base>;

  using Sacado::mpl::placeholders::_;
  using Manager = PHX::TemplateManager<EvalTypes,Base,Object<_>>;

  static_assert(Container::size == 3);
  static_assert(std::is_same_v<Container::object_type<Jacobian>,Object<Jacobian>>);
}

TEUCHOS_UNIT_TEST(template_container, build_concrete)
{
  Container c;
  TEST_ASSERT(!c.has<Residual>());
  c.build<Concrete>(std::string("eq"),2);

  TEST_ASSERT(c.has<Residual>() && c.has<Jacobian>() && c.has<Tangent>());
  Teuchos::RCP<Object<Jacobian>> j = c.get<Jacobian>();
  TEST_EQUALITY(j->describe(), "eq<J>");
  TEST_EQUALITY(j->order(), 2);
  TEST_EQUALITY(c.getAsBase<Tangent>()->describe(), "eq<T>");

  const Container& cc = c;
  Teuchos::RCP<const Object<Residual>> r = cc.get<Residual>();
  TEST_EQUALITY(r->order(), 2);
  Teuchos::RCP<const Base> rb = cc.getAsBase<Residual>();
  TEST_EQUALITY(rb.get(), static_cast<const Base*>(r.get()));
}

TEUCHOS_UNIT_TEST(template_container, build_lambda)
{
  Container c;
  int calls = 0;
  c.buildObjects([&]<typename EvalT>() -> Teuchos::RCP<Object<EvalT>> {
    ++calls;
    if constexpr (std::is_same_v<EvalT,Tangent>)
      return Teuchos::null;
    else
      return Teuchos::rcp(new Concrete<EvalT>("lambda",calls));
  });
  TEST_EQUALITY(calls, 3);
  TEST_EQUALITY(c.get<Residual>()->order(), 1);
  TEST_EQUALITY(c.get<Jacobian>()->order(), 2);
  TEST_ASSERT(!c.has<Tangent>());
}

TEUCHOS_UNIT_TEST(template_container, build_legacy)
{
  Container c;
  c.buildObjects(LegacyBuilder{4});
  TEST_EQUALITY(c.get<Tangent>()->describe(), "legacy<T>");
  TEST_EQUALITY(c.get<Tangent>()->order(), 4);

  // A build<T>() that returns a derived type is accepted at compile time.
  Container typed;
  typed.buildObjects(TypedLegacyBuilder{});
  TEST_EQUALITY(typed.get<Jacobian>()->describe(), "typed<J>");
  TEST_EQUALITY(typed.get<Jacobian>()->order(), 5);

  // A legacy builder's result is checked when it is built.
  Container bad;
  TEST_THROW(bad.buildObjects(WrongTypeBuilder{}), std::runtime_error);
}

TEUCHOS_UNIT_TEST(template_container, disable_and_reset)
{
  Container c;
  c.disable<Jacobian>();
  TEST_ASSERT(!c.isEnabled<Jacobian>());
  TEST_ASSERT(c.isEnabled<Residual>());
  c.build<Concrete>(std::string("eq"),1);
  TEST_ASSERT(c.has<Residual>());
  TEST_ASSERT(!c.has<Jacobian>());
  TEST_ASSERT(c.has<Tangent>());

  // disable() does not remove an existing object, reset() does.
  c.disable<Residual>();
  TEST_ASSERT(c.has<Residual>());
  c.reset<Residual>();
  TEST_ASSERT(!c.has<Residual>());

  c.set<Residual>(Teuchos::rcp(new Concrete<Residual>("set",9)));
  TEST_EQUALITY(c.get<Residual>()->order(), 9);
}

TEUCHOS_UNIT_TEST(template_container, for_each)
{
  Container c;
  c.disable<Jacobian>();
  c.build<Concrete>(std::string("eq"),1);

  std::vector<std::string> typed;
  c.forEach([&]<typename EvalT>(Object<EvalT>& obj) { typed.push_back(obj.describe()); });
  TEST_EQUALITY(typed.size(), 2);
  TEST_EQUALITY(typed[0], "eq<R>");
  TEST_EQUALITY(typed[1], "eq<T>");

  std::vector<std::string> base;
  const Container& cc = c;
  cc.forEachBase([&](const Base& b) { base.push_back(b.describe()); });
  TEST_EQUALITY(base.size(), 2);
  TEST_EQUALITY(base[1], "eq<T>");
}

TEUCHOS_UNIT_TEST(template_container, template_manager_interop)
{
  Manager tm;
  tm.disableType<Tangent>();
  tm.buildObjects(LegacyBuilder{3});

  // TemplateManager -> TemplateContainer shares the objects.
  auto c = Container::fromTemplateManager(tm);
  TEST_EQUALITY(c.get<Residual>().get(), tm.getAsObject<Residual>().get());
  TEST_EQUALITY(c.get<Jacobian>().get(), tm.getAsObject<Jacobian>().get());
  TEST_ASSERT(!c.has<Tangent>());

  // TemplateContainer -> TemplateManager shares the objects and the disabled types.
  Container c2;
  c2.disable<Residual>();
  c2.build<Concrete>(std::string("out"),5);
  Manager tm2;
  c2.toTemplateManager(tm2);
  TEST_ASSERT(tm2.getAsBase<Residual>().is_null());
  TEST_EQUALITY(tm2.getAsObject<Jacobian>().get(), c2.get<Jacobian>().get());
  TEST_EQUALITY(tm2.getAsObject<Tangent>()->describe(), "out<T>");

  // An object in the TemplateManager that is not an ObjectT<T> is rejected.
  Manager tm3;
  tm3.buildObjects(WrongTypeBuilder{});
  TEST_THROW(Container::fromTemplateManager(tm3), std::runtime_error);
}

// BaseT is optional.
TEUCHOS_UNIT_TEST(template_container, no_base)
{
  PHX::TemplateContainer<EvalTypes,Concrete> c;
  c.build(std::string("nb"),7);
  TEST_EQUALITY(c.get<Jacobian>()->describe(), "nb<J>");
}
