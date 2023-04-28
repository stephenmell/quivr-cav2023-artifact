import subprocess

all_tasks = (
    ('shibuya_one', 'east_to_north'),
    ('shibuya_one', 'south_to_east'),
    ('shibuya_one', 'west_to_south'),
    ('shibuya_one', 'north_to_west'),
    ('warsaw_one', 'eastward_3_high_accel'),
    ('warsaw_one', 'eastward_4_high_accel'),
    ('warsaw_two', 'southward_1_upper_westward_2_rightonred'),
    ('warsaw_two', 'eastward_3_passing_eastward_4'),
    ('warsaw_two', 'eastward_4_passing_eastward_3'),
    ('warsaw_two', 'parallel_eastward_4_eastward_3'),
    ('mabe22', 'approach'),
    ('mabe22', 'nose_nose_contact'),
    ('mabe22', 'nose_genital_contact'),
    ('mabe22', 'nose_ear_contact'),
    ('mabe22', 'chase'),
    ('mabe22', 'watching'),
    ('maritime_surveillance', 'a'),
)

datasets = []
for dataset_name, _ in all_tasks:
    if dataset_name not in datasets:
        datasets.append(dataset_name)

def sysrun(exec, *args):
    return subprocess.run([exec, *args], check = True)

torch_device = "cuda:0"

for dataset_name in datasets:
    cmdline = (
        "python",
        "lstm/train.py",
        dataset_name,
        torch_device,
    )
    print("STARTING:")
    print(*cmdline)
    sysrun(*cmdline)

skip_until = None

skip_still = skip_until is not None

for dataset_name, task_name in all_tasks:
    if (
        dataset_name,
        task_name,
    ) == skip_until:
        skip_still = False

    if skip_still:
        continue

    cmdline = (
        "python",
        "lstm/eval.py",
        dataset_name,
        task_name,
        torch_device,
    )
    print("STARTING:")
    print(*cmdline)
    sysrun(*cmdline)