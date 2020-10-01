TEST_CASE_ATTRIBUTE_NAME = "test_case"
TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME = "expected_value"


FN_NAME = ""
TEST_INPUTS = [
    {
        TEST_CASE_ATTRIBUTE_NAME: {

        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: None,
    },
]


def test(solution_class, test_inputs, fn_name):
    s = solution_class()
    for test_info in test_inputs:
        test_input = test_info[TEST_CASE_ATTRIBUTE_NAME]
        expected_output = test_info[TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME]

        actual_output = eval(f"s.{fn_name}(**{test_input})")

        pass_str = "Passed" if expected_output == actual_output else "FAILED"

        print(f"Expected output: {expected_output}\nActual output: {actual_output}\n{pass_str}\n")
