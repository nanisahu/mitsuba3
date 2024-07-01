# Syntax:  cmake -P variant-stub.cmake <variant> <input> <output>
# Replaces all occurrences of Dr.Jit prefix (e.g. drjit.llvm.ad) with 'mitsuba'

set(MI_STUB_REPLACE_PREFIX "drjit.")

if (CMAKE_ARGV3 MATCHES ".*cuda")
  string(APPEND MI_STUB_REPLACE_PREFIX "cuda")
elseif (CMAKE_ARGV3 MATCHES ".*llvm")
  string(APPEND MI_STUB_REPLACE_PREFIX "llvm")
else ()
  string(APPEND MI_STUB_REPLACE_PREFIX "scalar")
endif()

if (CMAKE_ARGV3 MATCHES ".*_ad")
  string(APPEND MI_STUB_REPLACE_PREFIX ".ad")
endif()

file(READ ${CMAKE_ARGV4} FILE_CONTENTS)
string(REPLACE ${MI_STUB_REPLACE_PREFIX} "mitsuba" FILE_CONTENTS "${FILE_CONTENTS}")
file(WRITE "${CMAKE_ARGV5}" "${FILE_CONTENTS}")
