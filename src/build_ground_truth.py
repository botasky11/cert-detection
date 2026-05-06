"""
构建 CMU-CERT r4.2 的恶意用户 ground truth。

清单来源: 官方 answers/insiders.csv (用户上传到 data/upload_files/insiders.csv)
经过滤 dataset == '4.2' 后, 共 70 个唯一恶意用户:
  Scenario 1 (30 users): 离职/不满员工, 上班期间 (常常下班后) 用 U 盘
                          带走专有数据并上传至 wikileaks.
  Scenario 2 (30 users): 即将跳槽的员工, 通过 jobsearch 找新工作,
                          然后窃取竞品资料并通过个人邮箱外发.
  Scenario 3 (10 users): 系统管理员通过键盘记录器获取 CEO 凭证后,
                          假冒 CEO 发送惊吓性邮件 (sabotage).

数据集中红队场景的官方完整名单 (经核对 insiders.csv).
"""

# 官方 insiders.csv 中 dataset='4.2' & scenario=1 的 30 个用户
SCENARIO_1_USERS = [
    'AAM0658', 'AJR0932', 'BDV0168', 'BIH0745', 'BLS0678',
    'BTL0226', 'CAH0936', 'DCH0843', 'EHB0824', 'EHD0584',
    'FMG0527', 'FTM0406', 'GHL0460', 'HJB0742', 'JMB0308',
    'JRG0207', 'KLH0596', 'KPC0073', 'LJR0523', 'LQC0479',
    'MAR0955', 'MAS0025', 'MCF0600', 'MYD0978', 'PPF0435',
    'RAB0589', 'RGG0064', 'RKD0604', 'TAP0551', 'WDD0366',
]

# 官方 insiders.csv 中 dataset='4.2' & scenario=2 的 30 个用户
SCENARIO_2_USERS = [
    'AAF0535', 'ABC0174', 'AKR0057', 'CCL0068', 'CEJ0109',
    'CQW0652', 'DIB0285', 'DRR0162', 'EDB0714', 'EGD0132',
    'FSC0601', 'HBO0413', 'HXL0968', 'IJM0776', 'IKR0401',
    'IUB0565', 'JJM0203', 'KRL0501', 'LCC0819', 'MDH0580',
    'MOS0047', 'NWT0098', 'PNL0301', 'PSF0133', 'RAR0725',
    'RHL0992', 'RMW0542', 'TNM0961', 'VSS0154', 'XHW0498',
]

# 官方 insiders.csv 中 dataset='4.2' & scenario=3 的 10 个用户
SCENARIO_3_USERS = [
    'BBS0039', 'BSS0369', 'CCA0046', 'CSC0217', 'GTD0219',
    'JGT0221', 'JLM0364', 'JTM0223', 'MPM0220', 'MSO0222',
]

# 三个场景两两不重叠, 共 70 个唯一用户. 后写覆盖前写, 但因为不重叠,
# 这里写法等价于"取并集 + 记录所属场景".
MALICIOUS_USERS = {}
for u in SCENARIO_1_USERS:
    MALICIOUS_USERS[u] = 1
for u in SCENARIO_2_USERS:
    MALICIOUS_USERS[u] = 2
for u in SCENARIO_3_USERS:
    MALICIOUS_USERS[u] = 3


def get_malicious_users():
    """返回恶意用户字典 {user_id: scenario_id}."""
    return dict(MALICIOUS_USERS)


def get_all_malicious_user_ids():
    """返回所有恶意用户 ID 集合."""
    return set(MALICIOUS_USERS.keys())


def load_from_official_csv(path='/home/user/webapp/data/upload_files/insiders.csv'):
    """
    可选: 从官方 insiders.csv 直接读取 (要求 pandas).
    用作单元测试与本文件硬编码列表的一致性校验.
    """
    import pandas as pd
    df = pd.read_csv(path, dtype={'dataset': str})
    r42 = df[df['dataset'] == '4.2']
    out = {}
    for _, row in r42.iterrows():
        out[row['user']] = int(row['scenario'])
    return out


if __name__ == "__main__":
    print(f"[Ground Truth] Scenario 1 users: {len(SCENARIO_1_USERS)}")
    print(f"[Ground Truth] Scenario 2 users: {len(SCENARIO_2_USERS)}")
    print(f"[Ground Truth] Scenario 3 users: {len(SCENARIO_3_USERS)}")
    print(f"[Ground Truth] Total unique malicious users: "
          f"{len(get_all_malicious_user_ids())}")

    # 与官方 CSV 的一致性自检
    try:
        official = load_from_official_csv()
        local = get_malicious_users()
        if official == local:
            print("[Ground Truth] ✅ 与官方 insiders.csv 完全一致.")
        else:
            print("[Ground Truth] ⚠ 与官方 insiders.csv 不一致!")
            o, l = set(official), set(local)
            print(f"  仅官方有: {sorted(o - l)}")
            print(f"  仅本地有: {sorted(l - o)}")
            for u in o & l:
                if official[u] != local[u]:
                    print(f"  场景不同 {u}: official={official[u]}, "
                          f"local={local[u]}")
    except Exception as e:
        print(f"[Ground Truth] (官方 CSV 一致性检查跳过: {e})")
