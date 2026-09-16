# Runs the tokenizer roundtrip when the released vocabulary is available, and prints a
# line ctest recognises as a skip when it is not.
#
# tiktoken.model has 154,820 entries and ships with the checkpoint, not with this
# repository, so this gate cannot run on a clean checkout. It is registered anyway: a
# check that vanishes from the list when it cannot run reads as "one fewer test to care
# about", and ctest's count silently stops matching what `make test` runs.
#
# GLM53F_TOK_FILES is read here, at TEST time rather than configure time, so
# `GLM53F_TOK_FILES=/path/to/GLM-5.3-Flash ctest` works without reconfiguring the build.

set(TOK_FILES "$ENV{GLM53F_TOK_FILES}")

if(TOK_FILES STREQUAL "")
  message("SKIPPED: GLM53F_TOK_FILES is not set. The vocabulary ships with the checkpoint;"
          " set GLM53F_TOK_FILES to a model directory to run this gate.")
  return()
endif()

if(NOT EXISTS "${TOK_FILES}/tiktoken.model")
  message("SKIPPED: no tiktoken.model in ${TOK_FILES}")
  return()
endif()

execute_process(COMMAND "${TEST_TOK}" "${TOK_FILES}" roundtrip "${INPUT}"
                RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "tokenizer roundtrip failed with status ${rc}")
endif()
