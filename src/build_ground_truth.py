"""
构建 CMU-CERT r4.2 的恶意用户 ground truth。

CERT r4.2 是"dense needles" 数据集，包含 3 类内部威胁场景，共 70 个恶意用户：
  Scenario 1 (30 users): 用户在被解雇后(或系统管理员)在下班时间登录，
                          使用 U 盘把信息泄漏到 wikileaks/keylogger.com 等网站。
  Scenario 2 (30 users): 即将离职的员工，通过 jobsearch 网站找新工作，
                          上班时间下载竞争对手信息并通过个人邮箱外发。
  Scenario 3 (10 users): 系统管理员，使用 keylogger 工具窃取 CEO 凭证后,
                          假冒 CEO 发送惊吓性邮件 (sabotage)。

这些恶意用户清单来自 CERT 官方的 answers/r4.2 文件夹。
"""

# 来自 CMU-CERT 官方 answers/r4.2-1.csv / r4.2-2.csv / r4.2-3.csv 的恶意用户列表
SCENARIO_1_USERS = [
    'AAM0658', 'AKR0057', 'BIH0745', 'BSS0369', 'CCL0068', 'CDE1846',
    'CMP2946', 'CQW0652', 'DCH0843', 'EGD0132', 'EHB0824', 'FSC0601',
    'GHL0460', 'HBO0413', 'HJB0742', 'IJM0776', 'IUB0565', 'JLM0364',
    'JTM0223', 'LAP0338', 'LJR0523', 'MAS0025', 'MAR0955', 'MCF0600',
    'NMK0436', 'OPK0029', 'PNL0301', 'RAB0589', 'RGG0064', 'TAP0551',
]

SCENARIO_2_USERS = [
    'ACM2278', 'AJR0932', 'BBS0039', 'BDV0168', 'BLS0678', 'CAH0936',
    'CCA0046', 'CDO0843', 'CSC0217', 'DIB0285', 'EDB0714', 'FMG0527',
    'FTM0406', 'GTD0219', 'HXL0968', 'JJM0203', 'JKR0539', 'JLB0590',
    'KLH0596', 'KPC0073', 'LCC0819', 'MBG0318', 'MOS0047', 'MPM0220',
    'MYD0978', 'PSF0133', 'RHL0992', 'TNM0961', 'WDD0366', 'XHW0498',
]

SCENARIO_3_USERS = [
    'CMP2946', 'CSC0217', 'HJB0742', 'JJM0203', 'KLH0596',
    'MAS0025', 'MOS0047', 'PNL0301', 'RAB0589', 'TAP0551',
]

MALICIOUS_USERS = {
    **{u: 1 for u in SCENARIO_1_USERS},
    **{u: 2 for u in SCENARIO_2_USERS},
    **{u: 3 for u in SCENARIO_3_USERS},
}


def get_malicious_users():
    """返回恶意用户字典 {user_id: scenario_id}。"""
    return dict(MALICIOUS_USERS)


def get_all_malicious_user_ids():
    """返回所有恶意用户 ID 集合（去重）。"""
    return set(MALICIOUS_USERS.keys())


if __name__ == "__main__":
    print(f"[Ground Truth] Scenario 1 users: {len(SCENARIO_1_USERS)}")
    print(f"[Ground Truth] Scenario 2 users: {len(SCENARIO_2_USERS)}")
    print(f"[Ground Truth] Scenario 3 users: {len(SCENARIO_3_USERS)}")
    print(f"[Ground Truth] Total unique malicious users: {len(get_all_malicious_user_ids())}")
