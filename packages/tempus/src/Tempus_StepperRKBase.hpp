// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#ifndef Tempus_StepperRKBase_hpp
#define Tempus_StepperRKBase_hpp

#include "Thyra_VectorBase.hpp"

#include "Tempus_Stepper.hpp"
#include "Tempus_StepperRKAppAction.hpp"
#include "Tempus_StepperRKModifierDefault.hpp"


namespace Tempus {

/** \brief Base class for Runge-Kutta methods, ExplicitRK, DIRK and IMEX.
 *
 *  Only common RK methods should be implemented in StepperRKBase.  All
 *  other Stepper methods should be implemented through Stepper,
 *  StepperExplicit or StepperImplicit.
 */
template<class Scalar>
class StepperRKBase : virtual public Tempus::Stepper<Scalar>
{

public:

  virtual int getStageNumber() const { return stageNumber_; }

  virtual void setStageNumber(int s) { stageNumber_ = s; }

  virtual Teuchos::RCP<Thyra::VectorBase<Scalar> > getStageX() {return stageX_;}

  virtual void setAppAction(Teuchos::RCP<StepperRKAppAction<Scalar> > appAction)
  {
    if (appAction == Teuchos::null) {
      // Create default appAction
      stepperRKAppAction_ =
        Teuchos::rcp(new StepperRKModifierDefault<Scalar>());
    } else {
      stepperRKAppAction_ = appAction;
    }
    this->isInitialized_ = false;
  }

  virtual Teuchos::RCP<StepperRKAppAction<Scalar> > getAppAction() const
    { return stepperRKAppAction_; }

protected:

  int stageNumber_;    //< The Runge-Kutta stage number, {0,...,s-1}.
  Teuchos::RCP<Thyra::VectorBase<Scalar> >  stageX_;
  Teuchos::RCP<StepperRKAppAction<Scalar> > stepperRKAppAction_;

};

} // namespace Tempus

#endif // Tempus_StepperRKBase_hpp
