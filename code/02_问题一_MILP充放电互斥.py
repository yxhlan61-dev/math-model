"""思路二：混合整数线性规划，使用二进制变量显式保证充放电互斥。"""

from problem1_solver_common import load_input, solve_milp, write_outputs


if __name__ == "__main__":
    input_data = load_input()
    milp_solution = solve_milp(input_data)
    _, validation = write_outputs(input_data, milp_solution)
    if validation["simultaneous_charge_discharge_count"]:
        raise SystemExit("MILP 互斥约束校验失败，请检查模型实现。")

