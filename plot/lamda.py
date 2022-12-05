import numpy as np
from matplotlib import pyplot as plt
import os

from scipy.interpolate import make_interp_spline
os.environ['OPENBLAS_NUM_THREADS'] = '1'
## 1
# lambda_value = [0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
# random = [0.95, 0.88, 0.8, 0.7, 0.5, 0.4, 0.25, 0.18, 0.1]
# adv_grad = [0.99, 0.92, 0.85, 0.79, 0.68, 0.64, 0.54, 0.46, 0.4]
# f = plt.figure()
# f.set_tight_layout(True)
# plt.plot(random,       label='Random')
# plt.plot(adv_grad, label='Gradient-guided')

# plt.legend()
# plt.xlabel('Ratio')
# plt.ylabel('ASR')
# plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=7)

# plt.savefig('Anomaly_0815.png')

## 2
#lambda_value = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#f = plt.figure()
#f.set_tight_layout(True)

#Anomoly = [4.3, 4.8, 5.2, 6.6, 7.3, 9.2, 10.8, 14.3, 15.2, 15.7]
#plt.plot(Anomoly)

# # plt.legend()
#plt.xlabel('Number')
#plt.ylabel('Anomaly Index')
#plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=7)

#plt.savefig('Anomaly_0815.png')

# # 3 trigger size
lambda_value = ['2x2', '4x4', '6x6', '8x8', '10x10', '12x12', '14x14']
f = plt.figure()
f.set_tight_layout(True)

APG = [1, 1, 0.9583, 0.8333, 0.75, 0.7083, 0.625]
NC = [1, 0.9167, 0.792, 0.8333, 0.598, 0.375, 0.125]
DF_TND = [1, 0.9583, 0.9167, 0.7917, 0.75, 0.7083, 0.5417]
plt.plot(APG, label='A2P (ours)', linewidth=4, linestyle='--', marker='o')
plt.plot(NC, label='NC', linewidth=4, linestyle=':', marker='v', mfc='white')
plt.plot(DF_TND, label='DF-TND', linewidth=4, linestyle='solid', marker='*', mfc='white')
plt.grid(True, color="gray",axis="both",ls="--",lw=1)
legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 14,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}
plt.legend(prop=legend_font, loc='lower left')
plt.xlabel('Trigger Size', fontsize=18, fontweight='bold')
plt.ylabel('ACC', fontsize=18, fontweight='bold')
plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)

plt.savefig('cvpr_images/trigger_size_1103.pdf', bbox_inches='tight')

# # 4  trigger transparency
lambda_value = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
f = plt.figure()
f.set_tight_layout(True)

# # f.spines["left"].set_visible(False)
# # f.spines["top"].set_visible(False)
# # f.spines["right"].set_visible(False)

APG = [0.6667, 0.7917, 0.875, 0.875, 0.9583, 1]
NC = [0.4167, 0.5833, 0.625, 0.9167, 1, 0.4583]
DF_TND = [0.5, 0.7917, 0.8333, 0.75, 0.7083, 0.7083]
plt.grid(True,color="gray",axis="both",ls="--",lw=1)
plt.plot(APG, label='A2P (ours)', linewidth=4, linestyle='--', marker='o')
plt.plot(NC, label='NC', linewidth=4, linestyle=':', marker='v', mfc='white')
plt.plot(DF_TND, label='DF-TND', linewidth=4, linestyle='solid', marker='*', mfc='white')
# # plt.grid(True,color="gray",axis="both",ls="--",lw=0.5)
legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 14,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}


plt.legend(prop=legend_font)
plt.xlabel('Trigger Transparency', fontsize=18, fontweight='bold')
plt.ylabel('ACC', fontsize=18, fontweight='bold')
plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)


plt.savefig('cvpr_images/trigger_transparency_1103.pdf',  bbox_inches='tight')


# # 4  region
lambda_value = ['2x2', '4x4', '8x8', '16x16', '32x32']
f = plt.figure()
f.set_tight_layout(True)
plt.grid(True, color="gray",axis="both",ls="--",lw=1)

clean = [1.7654, 1.3578, 1.6031, 1.5734, 1.2757]
badnet = [2.096, 2.567, 2.595, 3.027, 1.8371]
blend = [2.2354, 2.9678, 2.9562, 14.3861, 35.207]
plt.plot(clean,       label='Clean', linewidth=4, linestyle='--', marker='o', color='blue', mfc='white')
plt.plot(badnet, label='BadNet', linewidth=4,  linestyle='solid', marker='v', color='green')
plt.plot(blend,       label='Blend', linewidth=4, linestyle=':', marker='s', color='orange', mfc='white')
# # plt.plot(adv_grad, label='Gradient-guided')

legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 14,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}
plt.legend(prop=legend_font)
plt.xlabel('Region Size', fontsize=18, fontweight='bold')
plt.ylabel('Anomaly Index', fontsize=18, fontweight='bold')
plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
#plt.ylim(0, 20)

plt.savefig('cvpr_images/region_1103.pdf', bbox_inches='tight')


# # 6
lambda_value = ['2/255', '4/255', '8/255', '16/255', '32/255', '64/255']
f = plt.figure()
f.set_tight_layout(True)
plt.grid(True,color="gray",axis="both",ls="--",lw=1)
clean = [1.7654, 1.967, 1.6031, 1.6275, 1.4233, 2.0514]
badnet = [2.3543, 2.9675, 3.469, 4.601, 7.906, 20.299]
blend = [2.52, 2.698, 2.672, 2.622, 3.459, 3.986]
plt.plot(clean,       label='Clean', linewidth=4, linestyle='--', marker='o', color='blue', mfc='white')
plt.plot(badnet, label='BadNet', linewidth=4,  linestyle='solid', marker='v', color='green')
plt.plot(blend,       label='Blend', linewidth=4, linestyle=':', marker='s', mfc='white', color='orange')
# # plt.plot(adv_grad, label='Gradient-guided')

legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 14,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}
plt.legend(prop=legend_font)
plt.xlabel('Budget', fontsize=18, fontweight='bold')
plt.ylabel('Anomaly Index', fontsize=18, fontweight='bold')
plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
# # plt.ylim(0, 1)

plt.savefig('cvpr_images/budget_1103.pdf', bbox_inches='tight')


# # 6
# lambda_value = [100, 80, 40, 20, 10, 5]
# f = plt.figure()
# f.set_tight_layout(True)

# clean = [1.7654, 1.967, 1.6031, 1.6275, 1.4233, 2.8514]
# badnet = [2.3543, 2.9675, 3.4521, 2.1743, 2.7093, 10.4228]
# blend = [1.9256, 1.8769, 2.9562, 4.5244, 7.9467, 27.7651]
# plt.plot(clean,       label='Clean', linewidth=2.5, linestyle='solid')
# plt.plot(badnet, label='BadNet', linewidth=2.5,  linestyle='solid')
# plt.plot(blend,       label='Blend', linewidth=2.5, linestyle='solid')
# # plt.plot(adv_grad, label='Gradient-guided')

# legend_font = {
#     'family': 'Arial',  # 字体
#     'style': 'normal',
#     'size': 8,  # 字号
#     'weight': "bold",  # 是否加粗，不加粗
# }
# plt.legend(prop=legend_font)
# plt.xlabel('Budget', fontsize=12, fontweight='bold')
# plt.ylabel('Anomaly Index', fontsize=12, fontweight='bold')
# plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=10, fontweight='bold')
# plt.yticks(fontsize=10, fontweight='bold')
# # plt.ylim(0, 1)

# plt.savefig('budget_0815.png')

# 6
# lambda_value = ['1', '2', '3', '4', '5']
# f = plt.figure()
# f.set_tight_layout(True)
# plt.grid(True, color="gray",axis="both",ls="--",lw=0.5)
# random = [0.6, 0.725, 0.85, 0.8, 0.9]
# adv = [0.525, 0.6, 0.675, 0.575, 0.8]
# # blend = [1.9256, 1.8769, 2.9562, 4.5244, 7.9467, 27.7651]
# plt.plot(adv,       label='Random', linewidth=3, linestyle='--', color='blue', marker='s', mfc='white')
# plt.plot(random, label='Gradient-guided', linewidth=3,  linestyle='solid', color='red', marker='o', mfc='white')
# # plt.plot(blend,       label='Blend', linewidth=2.5, linestyle='solid')
# # plt.plot(adv_grad, label='Gradient-guided')

# legend_font = {
#     'family': 'Arial',  # 字体
#     'style': 'normal',
#     'size': 14,  # 字号
#     'weight': "bold",  # 是否加粗，不加粗
# }
# plt.legend(prop=legend_font)
# plt.xlabel('Step', fontsize=18, fontweight='bold')
# plt.ylabel('ACC', fontsize=18, fontweight='bold')
# plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.ylim(0, 1)

# plt.savefig('mask_study.png')


# 6
# lambda_value = ['1', '2', '3', '4', '5']
# f = plt.figure()
# f.set_tight_layout(True)
# plt.grid(True, color="gray",axis="both",ls="--",lw=0.5)
# random = [0.6, 0.725, 0.85, 0.8, 0.9]
# adv = [0.525, 0.6, 0.675, 0.575, 0.8]
# # blend = [1.9256, 1.8769, 2.9562, 4.5244, 7.9467, 27.7651]
# plt.plot(adv,       label='Random', linewidth=3, linestyle='--', color='blue', marker='s', mfc='white')
# plt.plot(random, label='Gradient-guided', linewidth=3,  linestyle='solid', color='red', marker='o', mfc='white')
# # plt.plot(blend,       label='Blend', linewidth=2.5, linestyle='solid')
# # plt.plot(adv_grad, label='Gradient-guided')

# legend_font = {
#     'family': 'Arial',  # 字体
#     'style': 'normal',
#     'size': 14,  # 字号
#     'weight': "bold",  # 是否加粗，不加粗
# }
# plt.legend(prop=legend_font)
# plt.xlabel('Step', fontsize=18, fontweight='bold')
# plt.ylabel('ACC', fontsize=18, fontweight='bold')
# plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.ylim(0, 1)

# plt.savefig('mask_study.png')

## xx
# lambda_value = ['0', '1', '2', '3', '4', '5']
# f = plt.figure()
# f.set_tight_layout(True)
# plt.grid(True, color="gray",axis="both",ls="--",lw=0.5)
# random = [1, 0.9065, 0.974, 0.989, 0.9955, 0.996]
# adv = [1, 0.9425, 0.9845, 0.9955, 0.9975, 0.997]

# random_b = [1, 0.947, 0.9065, 0.8465, 0.751, 0.6725]
# adv_b = [1, 0.9595, 0.9305, 0.8609, 0.7905, 0.7145]
# # blend = [1.9256, 1.8769, 2.9562, 4.5244, 7.9467, 27.7651]
# plt.plot(random,       label='Random (APG)', linewidth=3, linestyle='solid', color='b', marker='s', mfc='white')
# plt.plot(adv, label='Gradient-guided (APG)', linewidth=3,  linestyle='solid', color='r', marker='o', mfc='white')

# plt.plot(random_b,       label='Random (budget=8/255)', linewidth=3, linestyle='--', color='b', marker='s')
# plt.plot(adv_b, label='Gradient-guided (budget=8/255)', linewidth=3,  linestyle='--', color='r', marker='o')
# # plt.plot(blend,       label='Blend', linewidth=2.5, linestyle='solid')
# # plt.plot(adv_grad, label='Gradient-guided')

# legend_font = {
#     'family': 'Arial',  # 字体
#     'style': 'normal',
#     'size': 10,  # 字号
#     'weight': "bold",  # 是否加粗，不加粗
# }
# plt.legend(prop=legend_font)
# plt.xlabel('Step', fontsize=18, fontweight='bold')
# plt.ylabel('ASR', fontsize=18, fontweight='bold')
# plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.ylim(0.5, 1.1)

# plt.savefig('mask_study_for_ASR.png')


# ## xxxx
lambda_value = [1, 3, 5, 10, 20, 40, 80, 100]
f = plt.figure()
f.set_tight_layout(True)
plt.grid(True,color="gray",axis="both",ls="--",lw=1)
ACC = [0.375, 0.85, 0.9, 0.9, 0.925, 0.95, 0.925, 0.975]
# # badnet = [2.3543, 2.9675, 3.4521, 2.1743, 2.7093, 10.4228]
# # blend = [1.9256, 1.8769, 2.9562, 4.5244, 7.9467, 27.7651]
plt.plot(ACC, linewidth=4, linestyle='solid', color='green', marker='o', mfc='white')
# # plt.plot(badnet, label='BadNet', linewidth=2.5,  linestyle='solid')
# # plt.plot(blend,       label='Blend', linewidth=2.5, linestyle='solid')
# # plt.plot(adv_grad, label='Gradient-guided')

legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 14,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}
# # plt.legend(prop=legend_font)
plt.xlabel('Samples/Class', fontsize=18, fontweight='bold')
plt.ylabel('ACC', fontsize=18, fontweight='bold')
plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)

plt.savefig('cvpr_images/varying_samples.pdf', bbox_inches='tight')



## xxxx
#lambda_value = [1, 2, 3, 4, 5]
#f = plt.figure()
#f.set_tight_layout(True)
#plt.grid(True,color="g",axis="both",ls="--",lw=0.5)
#APG = [0.8, 0.5, 0.6, 0.7, 0.9]
#NC = [0.5, 0.9, 0.9, 1, 0.9]
# badnet = [2.3543, 2.9675, 3.4521, 2.1743, 2.7093, 10.4228]
# blend = [1.9256, 1.8769, 2.9562, 4.5244, 7.9467, 27.7651]
#plt.plot(APG, label='APG (ours)' , linewidth=3, linestyle='--', color='r', marker='o', mfc='white')
#plt.plot(NC, label='NC', linewidth=3,  linestyle='solid', color='b', marker='s', mfc='white')
# plt.plot(blend,       label='Blend', linewidth=2.5, linestyle='solid')
# plt.plot(adv_grad, label='Gradient-guided')

#legend_font = {
#    'family': 'Arial',  # 字体
#    'style': 'normal',
#    'size': 14,  # 字号
#    'weight': "bold",  # 是否加粗，不加粗
#}
#plt.legend(prop=legend_font)
#plt.xlabel('Triggers/Sample', fontsize=18, fontweight='bold')
#plt.ylabel('ACC', fontsize=18, fontweight='bold')
#plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=14, fontweight='bold')
#plt.yticks(fontsize=14, fontweight='bold')
#plt.ylim(0, 1.1)

#plt.savefig('multi_triggers.png')

## budget relation
#lambda_value = [1, 0.25, 0.17, 0.1, 0.04, 0.01]
#f = plt.figure()
#f.set_tight_layout(True)
#plt.grid(True,color="g",axis="both",ls="--",lw=0.5)
#EPS_8 = [0.9965, 0.7565, 0.6215, 0.4335, 0.21, 0.095]
#EPS_16 = [1, 0.95, 0.8575, 0.686, 0.3555, 0.125]
#EPS_32 = [1, 0.9965, 0.9795, 0.896, 0.548, 0.1925]
#EPS_64 = [1, 1, 1, 0.9995, 0.938, 0.4115]
#EPS_128 = [1, 1, 1, 1, 0.9425, 0.4275]
#EPS_255 = [1, 1, 1, 1, 0.985, 0.4775]
#plt.plot(EPS_8, label='budget=8/255' , linewidth=3, linestyle='--', color='r', marker='o', mfc='white')
#plt.plot(EPS_16, label='budget=16/255', linewidth=3,  linestyle='--', color='g', marker='s')
#plt.plot(EPS_32, label='budget=32/255', linewidth=3,  linestyle='solid', color='b', marker='v', mfc='white')
#plt.plot(EPS_64, label='budget=64/255', linewidth=3,  linestyle='solid', color='yellow', marker='*')
#plt.plot(EPS_128, label='budget=128/255', linewidth=3,  linestyle=':', color='black', marker='s', mfc='white')
#plt.plot(EPS_255, label='budget=255/255', linewidth=3,  linestyle=':', color='purple', marker='+')

#legend_font = {
#    'family': 'Arial',  # 字体
#    'style': 'normal',
#    'size': 14,  # 字号
#    'weight': "bold",  # 是否加粗，不加粗
#}
#plt.legend(prop=legend_font)
#plt.xlabel('Region Ratio', fontsize=18, fontweight='bold')
#plt.ylabel('ASR', fontsize=18, fontweight='bold')
#plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=14, fontweight='bold')
#plt.yticks(fontsize=14, fontweight='bold')
#plt.ylim(0, 1.1)

#plt.savefig('budget_relation.png')

## xx generatiom comparison for Badnets
#lambda_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#f = plt.figure()
#f.set_tight_layout(True)
#plt.grid(True, color="gray",axis="both",ls="--",lw=0.5)
#random_ASR = [1, 0.421, 0.593, 0.797, 0.933, 0.988, 0.998, 0.999, 1, 0.998, 0.934]
#grad_ASR = [1, 0.443, 0.646, 0.835, 0.953, 0.991, 0.997, 0.998, 0.998, 0.993, 0.948]

#random_ACC = [0.775, 0.35, 0.475, 0.45, 0.5, 0.65, 0.65, 0.6, 0.725, 0.975, 1]
#grad_ACC = [0.775, 0.45, 0.65, 0.6, 0.65, 0.85, 0.875, 0.9, 0.95, 0.975, 1]


#plt.plot(random_ACC,       label='Random', linewidth=3, linestyle=':', color='b', marker='s', mfc='white')
#plt.plot(grad_ACC, label='Gradient-guided', linewidth=3,  linestyle='solid', color='b', marker='o', mfc='white')

#legend_font = {
#    'family': 'Arial',  # 字体
#    'style': 'normal',
#    'size': 14,  # 字号
#    'weight': "bold",  # 是否加粗，不加粗
#}


#plt.xlabel('Stage', fontsize=18, fontweight='bold')
#plt.ylabel('ACC', fontsize=18, fontweight='bold', color='b')
#plt.tick_params(axis = 'y', labelcolor = 'b')
#plt.legend(loc = 'upper left', prop=legend_font)
#plt.ylim(0, 1.05)

#ax2 = plt.twinx()

#plt.plot(random_ASR,       label='Random', linewidth=3, linestyle=':', color='r', marker='s')
#plt.plot(grad_ASR, label='Gradient-guided', linewidth=3,  linestyle='solid', color='r', marker='o')

#ax2.set_ylabel('ASR', color = 'r')
#plt.tick_params(axis = 'y', labelcolor = 'r')
#plt.xlabel('Stage', fontsize=18, fontweight='bold')
#plt.ylabel('ASR', fontsize=18, fontweight='bold')


#plt.legend(loc = 'lower right', prop=legend_font)
#plt.ylim(0, 1.05)

#plt.savefig('generation_comparison.png')


## xx generatiom comparison for Badnets
lambda_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
f = plt.figure()
f.set_tight_layout(True)
plt.grid(True, color="gray",axis="both",ls="--",lw=1)
#random_ACC = [0.775, 0.35, 0.475, 0.45, 0.5, 0.65, 0.65, 0.6, 0.725, 0.975, 1]
#grad_ACC = [0.775, 0.45, 0.65, 0.6, 0.65, 0.85, 0.875, 0.9, 0.95, 0.975, 1]

random_ACC = [0.625, 0.575, 0.525, 0.475, 0.475, 0.55, 0.6, 0.625, 0.675, 0.775, 0.725]
grad_ACC = [0.625, 0.675, 0.65, 0.75, 0.95, 1, 1, 1, 1, 0.975, 1]

legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 14,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}

plt.plot(random_ACC,       label='Random', linewidth=4, linestyle='--', color='blue', marker='s')
plt.plot(grad_ACC, label='Gradient-guided', linewidth=4,  linestyle='solid', color='green', marker='o')

plt.xlabel('Stage', fontsize=18, fontweight='bold')
plt.ylabel('ACC', fontsize=18, fontweight='bold')

plt.legend(loc = 'lower left', prop=legend_font)

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)

plt.savefig('cvpr_images/generation_comparison_backdoor.pdf', bbox_inches='tight')

## generation comparison for Clean
lambda_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
f = plt.figure()
f.set_tight_layout(True)
plt.grid(True, color="gray",axis="both",ls="--",lw=0.5)
random_ASR = [0.996, 0.143, 0.225, 0.361, 0.535, 0.729, 0.876, 0.939, 0.968, 0.893, 0.403]
grad_ASR = [0.996, 0.154, 0.255, 0.402, 0.587, 0.771, 0.896, 0.946, 0.972, 0.924, 0.515]

random_ACC = [0.983, 0.933, 0.95, 1, 1, 1, 0.95, 0.933, 0.3, 0.133, 0.567]
grad_ACC = [0.983, 0.967, 0.967, 1, 0.983, 1, 0.983, 0.967, 0.867, 0.283, 0.783]


plt.plot(random_ACC,       label='Random', linewidth=3, linestyle=':', color='b', marker='s', mfc='white')
plt.plot(grad_ACC, label='Gradient-guided', linewidth=3,  linestyle='solid', color='b', marker='o', mfc='white')

legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 14,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}


plt.xlabel('Stage', fontsize=18, fontweight='bold')
plt.ylabel('ACC', fontsize=18, fontweight='bold', color='b')
plt.tick_params(axis = 'y', labelcolor = 'b')
plt.legend(loc = 'lower left', prop=legend_font)
#plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)

ax2 = plt.twinx()

#x_smooth = np.linspace(0, 10, 50)
#y3_smooth = make_interp_spline(lambda_value, random_ASR)(x_smooth)
#y4_smooth = make_interp_spline(lambda_value, grad_ASR)(x_smooth)
plt.plot(random_ASR,       label='Random', linewidth=3, linestyle=':', color='r', marker='s')
plt.plot(grad_ASR, label='Gradient-guided', linewidth=3,  linestyle='solid', color='r', marker='o')

ax2.set_ylabel('ASR-A', color = 'r')
plt.tick_params(axis = 'y', labelcolor = 'r')
plt.xlabel('Stage', fontsize=14, fontweight='bold')
plt.ylabel('ASR-A', fontsize=14, fontweight='bold')

#plt.xticks(ticks=range(len(lambda_value)), labels=lambda_value, fontsize=14, fontweight='bold')
#plt.yticks(fontsize=14, fontweight='bold')

plt.legend(loc = 'upper right', prop=legend_font)
plt.ylim(0, 1.05)

plt.savefig('generation_comparison_clean.png')


## generation comparison for Clean
lambda_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
f = plt.figure()
f.set_tight_layout(True)
plt.grid(True, color="gray",axis="both",ls="--",lw=1)
random_ASR = [0.996, 0.143, 0.225, 0.361, 0.535, 0.729, 0.876, 0.939, 0.968, 0.893, 0.403]
grad_ASR = [0.996, 0.154, 0.255, 0.402, 0.587, 0.771, 0.896, 0.946, 0.972, 0.924, 0.515]

#random_ACC = [0.983, 0.933, 0.95, 1, 1, 1, 0.95, 0.933, 0.3, 0.133, 0.567]
#grad_ACC = [0.983, 0.967, 0.967, 1, 0.983, 1, 0.983, 0.967, 0.867, 0.283, 0.783]

random_ACC = [0.975, 1, 0.975, 0.95, 0.95, 0.95, 0.975, 0.9, 0.85, 0.75, 0.725]
grad_ACC = [1, 1, 1, 0.95, 0.975, 1, 0.975, 0.925, 0.85, 0.9, 0.925]

plt.plot(random_ACC,       label='Random', linewidth=4, linestyle='--', color='blue', marker='s', mfc='white')
plt.plot(grad_ACC, label='Gradient-guided', linewidth=4,  linestyle='solid', color='green', marker='o', mfc='white')

legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 14,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}


plt.legend(prop=legend_font)

plt.xlabel('Stage', fontsize=18, fontweight='bold')
plt.ylabel('ACC', fontsize=18, fontweight='bold')

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)

plt.savefig('cvpr_images/generation_comparison_clean.pdf', bbox_inches='tight')




## budget schedule for clean
lambda_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
f = plt.figure()
f.set_tight_layout(True)
plt.grid(True, color="gray",axis="both",ls="--",lw=1)

A2P = [1, 1, 1, 0.95, 0.975, 1, 0.975, 0.925, 0.85, 0.9, 0.925]
BOX = [1, 1, 1, 1, 0.975, 1, 1, 0.875, 0.85, 0.775, 0.725]
SPARSITY = [1, 1, 0.025, 0.8, 0.55, 0.25, 0.675, 0.9, 0.925, 0.7, 0.8]

plt.plot(BOX,       label='Box', linewidth=4, linestyle='--', color='blue', marker='s', mfc='white')
plt.plot(A2P, label='Box-to-Sparsity', linewidth=4,  linestyle='solid', color='green', marker='o')
plt.plot(SPARSITY, label='Sparsity', linewidth=4,  linestyle=':', color='orange', marker='*',  mfc='white')


legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 14,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}

plt.legend(prop=legend_font)

plt.xlabel('Stage', fontsize=18, fontweight='bold')
plt.ylabel('ACC', fontsize=18, fontweight='bold')

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)

plt.savefig('cvpr_images/schedule_comparison_clean.pdf', bbox_inches='tight')


## budget schdule for badnet
lambda_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
f = plt.figure()
f.set_tight_layout(True)
plt.grid(True, color="gray",axis="both",ls="--",lw=1)

A2P = [0.625, 0.675, 0.65, 0.75, 0.95, 1, 1, 1, 1, 0.975, 1]
BOX = [0.575, 0.6, 0.625, 0.625, 0.85, 1, 0.975, 0.975, 0.95, 0.9, 0.9]
SPARSITY = [0.675, 0.7, 0.55, 0.775, 1, 0.975, 1, 1, 1, 1, 1]

plt.plot(BOX,       label='Box', linewidth=4, linestyle='--', color='blue', marker='s', mfc='white')
plt.plot(A2P, label='Box-to-Sparsity', linewidth=4,  linestyle='solid', color='green', marker='o')
plt.plot(SPARSITY, label='Sparsity', linewidth=4,  linestyle=':', color='orange', marker='*',  mfc='white')

legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 14,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}

plt.legend(prop=legend_font)

plt.xlabel('Stage', fontsize=18, fontweight='bold')
plt.ylabel('ACC', fontsize=18, fontweight='bold')

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)

plt.savefig('cvpr_images/schedule_comparison_backdoor.pdf', bbox_inches='tight')
