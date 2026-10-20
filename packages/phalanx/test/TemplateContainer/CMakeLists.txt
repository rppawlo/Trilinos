
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  template_container
  SOURCES TemplateContainerTest.cpp
  TESTONLYLIBS phalanx_unit_test_main
  COMM serial mpi
  NUM_MPI_PROCS 1
)
