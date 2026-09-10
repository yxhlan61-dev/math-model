"""思路一：连续线性规划直接求解，求解后核查是否同时充放电。"""

from problem1_solver_common import load_input, solve_lp, write_outputs


if __name__ == "__main__":
    input_data = load_input()
    lp_solution = solve_lp(input_data)
    _, validation = write_outputs(input_data, lp_solution)
    if validation["simultaneous_charge_discharge_count"]:
        raise SystemExit("LP 结果出现同时充放电，请查看校验摘要并采用 MILP 结果。")

