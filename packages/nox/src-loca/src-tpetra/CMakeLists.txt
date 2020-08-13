SET(HEADERS "")
SET(SOURCES "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# Base library
APPEND_SET(HEADERS
  NOX_TpetraTypedefs.hpp
  LOCA_Tpetra_Factory.hpp
  LOCA_BorderedSolver_TpetraHouseholder.hpp
  )

APPEND_SET(SOURCES
  LOCA_Tpetra_Factory.cpp
  LOCA_BorderedSolver_TpetraHouseholder.cpp
  )

IF (NOX_ENABLE_ABSTRACT_IMPLEMENTATION_EPETRA AND NOX_ENABLE_LOCA)
  TRIBITS_ADD_LIBRARY(
    locatpetra
    HEADERS ${HEADERS}
    SOURCES ${SOURCES}
    DEPLIBS loca nox
    )
ENDIF()
