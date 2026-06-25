import argparse
import glob
import os

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions


def create_pose_header():
    dimensions = PoseHeaderDimensions(width=1000, height=1000, depth=1000)
    components = []
    for hand in ["left_hand", "right_hand"]:
        points = [f"{hand}_{i}" for i in range(21)]
        components.append(
            PoseHeaderComponent(name=hand, points=points, limbs=[], colors=[], point_format="XYZC")
        )
    return PoseHeader(version=0.1, dimensions=dimensions, components=components)


def convert_to_pose(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    header = create_pose_header()
    classes = sorted(
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    )

    for cls in classes:
        cls_input = os.path.join(input_dir, cls)
        cls_output = os.path.join(output_dir, cls)
        os.makedirs(cls_output, exist_ok=True)

        subdirs = [
            d for d in os.listdir(cls_input) if os.path.isdir(os.path.join(cls_input, d))
        ]

        if subdirs:
            for seq_dir in subdirs:
                seq_path = os.path.join(cls_input, seq_dir)
                out_path = os.path.join(cls_output, f"{seq_dir}.pose")

                frame_files = glob.glob(os.path.join(seq_path, "*.npy")) + glob.glob(
                    os.path.join(seq_path, "*.np")
                )
                if not frame_files:
                    continue

                frame_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
                sequence_data = []
                for frame_file in frame_files:
                    try:
                        frame_data = np.load(frame_file)
                        if frame_data.shape == (126,):
                            frame_data = frame_data.reshape(42, 3)
                        sequence_data.append(frame_data)
                    except Exception as e:
                        print(f"Error loading {frame_file}: {e}")

                if not sequence_data:
                    continue

                data = np.stack(sequence_data)
                if len(data.shape) == 3 and data.shape[1:] == (42, 3):
                    data = np.expand_dims(data, axis=1)

                conf = np.ones(data.shape[:-1], dtype=np.float32)
                body = NumPyPoseBody(fps=30, data=data.astype(np.float32), confidence=conf)
                pose = Pose(header, body)
                with open(out_path, "wb") as f:
                    pose.write(f)
        else:
            npy_files = glob.glob(os.path.join(cls_input, "*.npy")) + glob.glob(
                os.path.join(cls_input, "*.np")
            )
            for npy_file in npy_files:
                file_name = os.path.basename(npy_file).split(".")[0]
                out_path = os.path.join(cls_output, f"{file_name}.pose")
                try:
                    data = np.load(npy_file)
                    if data.shape[-1] == 126:
                        data = data.reshape(data.shape[0], 42, 3)
                except Exception as e:
                    print(f"Error loading {npy_file}: {e}")
                    continue

                if len(data.shape) == 3 and data.shape[1:] == (42, 3):
                    data = np.expand_dims(data, axis=1)

                conf = np.ones(data.shape[:-1], dtype=np.float32)
                body = NumPyPoseBody(fps=30, data=data.astype(np.float32), confidence=conf)
                pose = Pose(header, body)
                with open(out_path, "wb") as f:
                    pose.write(f)

        print(f"Converted class {cls}")


def main():
    parser = argparse.ArgumentParser(description="Convert ISL landmark .npy data to .pose format")
    parser.add_argument("--input-dir", required=True, help="Raw ISL dataset root (A-Z folders)")
    parser.add_argument("--output-dir", required=True, help="Output directory for .pose files")
    args = parser.parse_args()

    print("Starting conversion...")
    convert_to_pose(args.input_dir, args.output_dir)
    print("Conversion complete.")


if __name__ == "__main__":
    main()
