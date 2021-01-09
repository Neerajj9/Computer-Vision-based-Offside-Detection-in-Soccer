#pragma once
#ifndef ANDRES_ILP_GUROBI_HXX
#define ANDRES_ILP_GUROBI_HXX

#include <limits>

#include "gurobi_c++.h"

namespace andres {
namespace ilp {

class Gurobi {
public:
    enum PreSolver {PRE_SOLVER_AUTO, PRE_SOLVER_PRIMAL, PRE_SOLVER_DUAL, PRE_SOLVER_NONE};
    enum LPSolver {LP_SOLVER_PRIMAL_SIMPLEX, LP_SOLVER_DUAL_SIMPLEX, LP_SOLVER_BARRIER, LP_SOLVER_SIFTING};

    Gurobi();
    ~Gurobi();
    void setNumberOfThreads(const size_t);
    void setAbsoluteGap(const double);
    void setRelativeGap(const double);
    void setVerbosity(const bool);
    void setLPSolver(const LPSolver);
    void setPreSolver(const PreSolver, const int = -1);
    void initModel(const size_t, const double*);
    template<class Iterator>
        void setStart(Iterator);
    template<class VariableIndexIterator, class CoefficientIterator>
        void addConstraint(VariableIndexIterator, VariableIndexIterator,
                           CoefficientIterator, const double, const double);
    void optimize();

    double label(const size_t) const;
    size_t numberOfThreads() const;
    double absoluteGap() const;
    double relativeGap() const;

private:
    GRBEnv gurobiEnvironment_;
    GRBModel* gurobiModel_;
    GRBVar* gurobiVariables_;
    GRBLinExpr gurobiObjective_;
    size_t nVariables_;
};

inline
Gurobi::Gurobi() :
    gurobiEnvironment_(),
    gurobiModel_(NULL),
    gurobiVariables_(NULL),
    gurobiObjective_(),
    nVariables_(0)
{
    setVerbosity(false);
}

inline
Gurobi::~Gurobi() {
    if (gurobiModel_ != NULL)
        delete gurobiModel_;

    if (gurobiVariables_ != NULL)
        delete[] gurobiVariables_;
}

inline void
Gurobi::setNumberOfThreads(
    const size_t numberOfThreads
) {
    gurobiEnvironment_.set(GRB_IntParam_Threads, numberOfThreads);
}

inline void
Gurobi::setAbsoluteGap(
    const double gap
) {
    gurobiEnvironment_.set(GRB_DoubleParam_MIPGapAbs, gap);
}

inline void
Gurobi::setRelativeGap(
    const double gap
) {
    gurobiEnvironment_.set(GRB_DoubleParam_MIPGap, gap);
}

inline void
Gurobi::setVerbosity(
    const bool verbosity
) {
    if(verbosity) {
        gurobiEnvironment_.set(GRB_IntParam_OutputFlag, 1);
    }
    else {
        gurobiEnvironment_.set(GRB_IntParam_OutputFlag, 0);
    }
}

inline void
Gurobi::setPreSolver(
    const PreSolver preSolver,
    const int passes
) {
    switch(preSolver) {
    case PRE_SOLVER_NONE:
        gurobiEnvironment_.set(GRB_IntParam_Presolve, 0);
        return;
    case PRE_SOLVER_AUTO:
        gurobiEnvironment_.set(GRB_IntParam_PreDual, -1);
        break;
    case PRE_SOLVER_PRIMAL:
        gurobiEnvironment_.set(GRB_IntParam_PreDual, 0);
        break;
    case PRE_SOLVER_DUAL:
        gurobiEnvironment_.set(GRB_IntParam_PreDual, 1);
        break;
    }
    gurobiEnvironment_.set(GRB_IntParam_PrePasses, passes);

    // crushing allows the solver to translate variable indices in cuts 
    // to variable indices of the pre-solved problem
    /*
    if(crush) {
        gurobiEnvironment_.set(GRB_IntParam_PreCrush, 1);
    }
    else {
        gurobiEnvironment_.set(GRB_IntParam_PreCrush, 0);
    }
    */
}

inline void
Gurobi::setLPSolver(
    const LPSolver lpSolver
) {
    switch(lpSolver) {
    case LP_SOLVER_PRIMAL_SIMPLEX:
        gurobiEnvironment_.set(GRB_IntParam_NodeMethod, 0);
        break;
    case LP_SOLVER_DUAL_SIMPLEX:
        gurobiEnvironment_.set(GRB_IntParam_NodeMethod, 1);
        break;
    case LP_SOLVER_BARRIER:
        gurobiEnvironment_.set(GRB_IntParam_NodeMethod, 2);
        break;
    case LP_SOLVER_SIFTING:
        gurobiEnvironment_.set(GRB_IntParam_NodeMethod, 1); // dual simplex
        gurobiEnvironment_.set(GRB_IntParam_SiftMethod, 1); // moderate, 2 = aggressive
        break;
    }
}

inline void
Gurobi::initModel(
    const size_t numberOfVariables,
    const double* coefficients
) {
    nVariables_ = numberOfVariables;
    
    if (gurobiModel_ != NULL)
        delete gurobiModel_;

    if (gurobiVariables_ != NULL)
        delete[] gurobiVariables_;

    gurobiModel_ = new GRBModel(gurobiEnvironment_);
    gurobiVariables_ = gurobiModel_->addVars(numberOfVariables, GRB_BINARY);
    gurobiModel_->update();
    gurobiObjective_.addTerms(coefficients, gurobiVariables_, numberOfVariables);
    gurobiModel_->setObjective(gurobiObjective_);
}

inline void
Gurobi::optimize() {
    gurobiModel_->optimize();
}

inline double
Gurobi::label(
    const size_t variableIndex
) const {
    return gurobiVariables_[variableIndex].get(GRB_DoubleAttr_X);
}

inline size_t
Gurobi::numberOfThreads() const {
    return gurobiEnvironment_.get(GRB_IntParam_Threads);
}

inline double
Gurobi::absoluteGap() const {
    return gurobiEnvironment_.get(GRB_DoubleParam_MIPGapAbs);
}

inline double
Gurobi::relativeGap() const {
    return gurobiEnvironment_.get(GRB_DoubleParam_MIPGap);
}

template<class VariableIndexIterator, class CoefficientIterator>
inline void
Gurobi::addConstraint(
    VariableIndexIterator viBegin,
    VariableIndexIterator viEnd,
    CoefficientIterator coefficient,
    const double lowerBound,
    const double upperBound
) {
    GRBLinExpr expression;
    for(; viBegin != viEnd; ++viBegin, ++coefficient) {
        expression += (*coefficient) * gurobiVariables_[static_cast<size_t>(*viBegin)];
    }
    if(lowerBound == upperBound) {
        GRBLinExpr exact(lowerBound);
        gurobiModel_->addConstr(expression, GRB_EQUAL, exact);
    }
    else {
        if(lowerBound != -std::numeric_limits<double>::infinity()) {
            GRBLinExpr lower(lowerBound);
            gurobiModel_->addConstr(expression, GRB_GREATER_EQUAL, lower);
        }
        if(upperBound != std::numeric_limits<double>::infinity()) {
            GRBLinExpr upper(upperBound);
            gurobiModel_->addConstr(expression,GRB_LESS_EQUAL, upper);
        }
    }
}

template<class Iterator>
inline void
Gurobi::setStart(
    Iterator valueIterator
) {
    for(size_t j = 0; j < nVariables_; ++j, ++valueIterator) {
        gurobiVariables_[j].set(GRB_DoubleAttr_Start, static_cast<double>(*valueIterator));
    }
}

} // namespace ilp
} // namespace andres

#endif // #ifndef ANDRES_ILP_GUROBI_HXX
