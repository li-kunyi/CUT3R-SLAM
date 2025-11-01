#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def convert_c2w_to_w2c(input_txt, output_txt):
    # 读取数据
    data = np.loadtxt(input_txt)
    out_lines = []

    for row in data:
        tstamp = row[0]
        tx, ty, tz = row[1:4]
        qx, qy, qz, qw = row[4:8]

        # 构造旋转和平移 (c2w)
        R_c2w = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t_c2w = np.array([tx, ty, tz]).reshape(3, 1)

        # 构造齐次矩阵
        T_c2w = np.eye(4)
        T_c2w[:3, :3] = R_c2w
        T_c2w[:3, 3:] = t_c2w

        # 求逆 (w2c)
        T_w2c = np.linalg.inv(T_c2w)

        R_w2c = T_w2c[:3, :3]
        t_w2c = T_w2c[:3, 3]

        # 转回四元数 (scipy 输出顺序是 [x, y, z, w])
        q_w2c = R.from_matrix(R_w2c).as_quat()

        # 拼接结果
        out_line = [tstamp, t_w2c[0], t_w2c[1], t_w2c[2],
                    q_w2c[0], q_w2c[1], q_w2c[2], q_w2c[3]]
        out_lines.append(out_line)

    # 保存
    np.savetxt(output_txt, np.array(out_lines),
               fmt="%.6f %.7f %.7f %.7f %.7f %.7f %.7f %.7f")


def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectory from c2w to w2c and save as txt")
    parser.add_argument("--input", help="Input trajectory txt (c2w format)")
    parser.add_argument("--output", help="Output trajectory txt (w2c format)")
    args = parser.parse_args()

    convert_c2w_to_w2c(args.input, args.output)
    print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
