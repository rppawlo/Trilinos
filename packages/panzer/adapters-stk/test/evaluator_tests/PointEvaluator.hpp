// @HEADER
// *****************************************************************************
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//
// Copyright 2011 NTESS and the Panzer contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef POINT_EVALUATOR
#define POINT_EVALUATOR

#include <string>
#include "Phalanx_Evaluator_Macros.hpp"
#include "Phalanx_MDField.hpp"
#include "Panzer_IntegrationRule.hpp"
#include "Panzer_IntegrationDescriptor.hpp"
#include "Panzer_Workset_Utilities.hpp"
#include "Panzer_Evaluator_Macros.hpp"

template <typename ScalarT>
class PointEvaluation {
public:
   // virtual void evaluateContainer(const Kokkos::DynRankView<double,PHX::Device> & points,
   virtual void evaluateContainer(const PHX::MDField<const ScalarT,panzer::Cell,panzer::IP,panzer::Dim> & points,
                                  PHX::MDField<ScalarT> & field) const = 0;
};

template<typename EvalT, typename Traits>
class PointEvaluator
  :
  public panzer::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{
  public:

    PointEvaluator(
      const Teuchos::ParameterList& p);

    void
    evaluateFields(
      typename Traits::EvalData d);

  private:

    using ScalarT = typename EvalT::ScalarT;

  PHX::MDField<ScalarT> scalar;
  PHX::MDField<ScalarT> vectorField;

  bool isVector;
  panzer::IntegrationDescriptor integration_descriptor;
  Teuchos::RCP<const PointEvaluation<ScalarT> > function;
}; // end of class PointEvaluator


//**********************************************************************
template<typename EvalT, typename Traits>
PointEvaluator<EvalT, Traits>::
PointEvaluator(
  const Teuchos::ParameterList& p)
{
  // Read from parameters
  const std::string name = p.get<std::string>("Name");
  Teuchos::RCP<panzer::IntegrationRule> ir
     = p.get< Teuchos::RCP<panzer::IntegrationRule> >("IR");
  isVector = p.get<bool>("Is Vector");
  function = p.get<Teuchos::RCP<const PointEvaluation<ScalarT> > >("Point Evaluator");

  integration_descriptor = *ir;

  // grab information from quadrature rule
  if(isVector) {
     vectorField = PHX::MDField<ScalarT>(name, ir->dl_vector);
     this->addEvaluatedField(vectorField);
  }
  else {
     scalar = PHX::MDField<ScalarT>(name, ir->dl_scalar);
     this->addEvaluatedField(scalar);
  }


  std::string n = "PointEvaluator: " + name;
  this->setName(n);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void
PointEvaluator<EvalT, Traits>::
evaluateFields(
  typename Traits::EvalData workset)
{
   const auto cubature_points = this->wda(workset)
                                .getIntegrationValues(integration_descriptor, panzer::useLazyWorksetEvaluation())
                                .getCubaturePoints();
   if(isVector)
      function->evaluateContainer(cubature_points,vectorField);
   else
      function->evaluateContainer(cubature_points,scalar);
}

//**********************************************************************

#endif
