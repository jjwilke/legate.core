function(legate_test name)
  set(test_name "${name}_test")
  set(SOURCES "${test_name}.cc")
  add_executable(${test_name} ${SOURCES})
  target_link_libraries(${test_name} legion::testutils)
  add_test(NAME ${name} COMMAND ${test_name})
endfunction()


legate_test(instance_manager)
