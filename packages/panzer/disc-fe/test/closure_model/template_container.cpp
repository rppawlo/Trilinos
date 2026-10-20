// @HEADER
// *****************************************************************************
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//
// Copyright 2011 NTESS and the Panzer contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teuchos_UnitTestHarness.hpp"

// Each container header must compile on its own.
#include "Panzer_EquationSet_TemplateContainer.hpp"
#include "Panzer_BCStrategy_TemplateContainer.hpp"
#include "Panzer_ClosureModel_Factory_TemplateContainer.hpp"
#include "Panzer_AssemblyEngine_TemplateContainer.hpp"
#include "Panzer_ResponseEvaluatorFactory_TemplateContainer.hpp"

#include "Panzer_ClosureModel_Factory_TemplateManager.hpp"
#include "user_app_ClosureModel_Factory_TemplateBuilder.hpp"

#include <type_traits>

namespace panzer {

  using T = panzer::Traits;

  static_assert(std::is_same_v<EquationSet_TemplateContainer<T>::object_type<T::Jacobian>, EquationSet<T::Jacobian>>);
  static_assert(std::is_same_v<EquationSet_TemplateContainer<T>::base_type, EquationSetBase>);
  static_assert(std::is_same_v<BCStrategy_TemplateContainer<T>::object_type<T::Residual>, BCStrategy<T::Residual>>);
  static_assert(std::is_same_v<BCStrategy_TemplateContainer<T>::base_type, BCStrategyBase>);
  static_assert(std::is_same_v<ClosureModelFactory_TemplateContainer<T>::object_type<T::Tangent>, ClosureModelFactory<T::Tangent>>);
  static_assert(std::is_same_v<ClosureModelFactory_TemplateContainer<T>::base_type, ClosureModelFactoryBase>);
  static_assert(std::is_same_v<AssemblyEngine_TemplateContainer<T>::object_type<T::Residual>, AssemblyEngine<T::Residual>>);
  static_assert(std::is_same_v<AssemblyEngine_TemplateContainer<T>::base_type, Base>);
  static_assert(std::is_same_v<ResponseEvaluatorFactory_TemplateContainer<T>::object_type<T::Jacobian>, ResponseEvaluatorFactory<T::Jacobian>>);
  static_assert(std::is_same_v<ResponseEvaluatorFactory_TemplateContainer<T>::base_type, ResponseEvaluatorFactoryBase>);
  static_assert(EquationSet_TemplateContainer<T>::size == PHX::mpl::size_v<T::EvalTypes>);

  TEUCHOS_UNIT_TEST(template_container, closure_model_factory_legacy_builder)
  {
    user_app::MyModelFactory_TemplateBuilder builder;

    ClosureModelFactory_TemplateContainer<T> c;
    c.disable<T::Tangent>();
    c.buildObjects(builder);

    TEST_ASSERT(c.has<T::Residual>());
    TEST_ASSERT(c.has<T::Jacobian>());
    TEST_ASSERT(!c.has<T::Tangent>());

    Teuchos::RCP<ClosureModelFactory<T::Residual>> r = c.get<T::Residual>();
    TEST_ASSERT(Teuchos::nonnull(Teuchos::rcp_dynamic_cast<user_app::MyModelFactory<T::Residual>>(r)));
    Teuchos::RCP<ClosureModelFactoryBase> rb = c.getAsBase<T::Residual>();
    TEST_EQUALITY(rb.get(), static_cast<ClosureModelFactoryBase*>(r.get()));

    int count = 0;
    c.forEachBase([&](ClosureModelFactoryBase&) { ++count; });
    TEST_EQUALITY(count, 2);
  }

  TEUCHOS_UNIT_TEST(template_container, closure_model_factory_template_manager_interop)
  {
    user_app::MyModelFactory_TemplateBuilder builder;

    ClosureModelFactory_TemplateManager<T> tm;
    tm.buildObjects(builder);

    auto c = ClosureModelFactory_TemplateContainer<T>::fromTemplateManager(tm);
    TEST_EQUALITY(c.get<T::Residual>().get(), tm.getAsObject<T::Residual>().get());
    TEST_EQUALITY(c.get<T::Jacobian>().get(), tm.getAsObject<T::Jacobian>().get());
    TEST_EQUALITY(c.get<T::Tangent>().get(), tm.getAsObject<T::Tangent>().get());

    ClosureModelFactory_TemplateManager<T> tm2;
    c.toTemplateManager(tm2);
    TEST_EQUALITY(tm2.getAsObject<T::Jacobian>().get(), c.get<T::Jacobian>().get());
  }

}
