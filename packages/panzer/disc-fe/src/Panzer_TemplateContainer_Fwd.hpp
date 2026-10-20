// @HEADER
// *****************************************************************************
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//
// Copyright 2011 NTESS and the Panzer contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef PANZER_TEMPLATE_CONTAINER_FWD_HPP
#define PANZER_TEMPLATE_CONTAINER_FWD_HPP

#include "Phalanx_TemplateContainer_Fwd.hpp"

/** \file Panzer_TemplateContainer_Fwd.hpp
    \brief Declares the Panzer PHX::TemplateContainer aliases without
    including their definitions. Include the matching
    Panzer_*_TemplateContainer.hpp header to use one.

    These replace the Panzer *_TemplateManager classes.
*/

namespace panzer {

  class Base;
  class EquationSetBase;
  class BCStrategyBase;
  class ClosureModelFactoryBase;
  class ResponseEvaluatorFactoryBase;

  template <typename EvalT> class EquationSet;
  template <typename EvalT> class BCStrategy;
  template <typename EvalT> class ClosureModelFactory;
  template <typename EvalT> class AssemblyEngine;
  template <typename EvalT> class ResponseEvaluatorFactory;

  //! One panzer::EquationSet<EvalT> per evaluation type. Replaces EquationSet_TemplateManager.
  template <typename Traits>
  using EquationSet_TemplateContainer =
    PHX::TemplateContainer<typename Traits::EvalTypes, EquationSet, EquationSetBase>;

  //! One panzer::BCStrategy<EvalT> per evaluation type. Replaces BCStrategy_TemplateManager.
  template <typename Traits>
  using BCStrategy_TemplateContainer =
    PHX::TemplateContainer<typename Traits::EvalTypes, BCStrategy, BCStrategyBase>;

  //! One panzer::ClosureModelFactory<EvalT> per evaluation type. Replaces ClosureModelFactory_TemplateManager.
  template <typename Traits>
  using ClosureModelFactory_TemplateContainer =
    PHX::TemplateContainer<typename Traits::EvalTypes, ClosureModelFactory, ClosureModelFactoryBase>;

  //! One panzer::AssemblyEngine<EvalT> per evaluation type. Replaces AssemblyEngine_TemplateManager.
  template <typename Traits>
  using AssemblyEngine_TemplateContainer =
    PHX::TemplateContainer<typename Traits::EvalTypes, AssemblyEngine, Base>;

  //! One panzer::ResponseEvaluatorFactory<EvalT> per evaluation type. Replaces ResponseEvaluatorFactory_TemplateManager.
  template <typename Traits>
  using ResponseEvaluatorFactory_TemplateContainer =
    PHX::TemplateContainer<typename Traits::EvalTypes, ResponseEvaluatorFactory, ResponseEvaluatorFactoryBase>;

}

#endif
