
import os, glob

cancer = os.listdir("/ssd_scratch/cvit/medicalImaging/PATCHES_KICH/cancer")
normal = os.listdir("/ssd_scratch/cvit/medicalImaging/PATCHES_KICH/normal")

l_c = len(cancer)
l_n = len(normal)

train_c, valid_c, test_c = cancer[:int(l_c*.71)], cancer[int(l_c*.71):int(l_c*.86)], cancer[int(l_c*.86):]
train_n, valid_n, test_n = normal[:50], normal[50:61], normal[61:]

open("KICH_cancer_train.txt", 'w').write('\n'.join(train_c))
open("KICH_cancer_valid.txt", 'w').write('\n'.join(valid_c))
open("KICH_cancer_test.txt", 'w').write('\n'.join(test_c))
open("KICH_normal_train.txt", 'w').write('\n'.join(train_n))
open("KICH_normal_valid.txt", 'w').write('\n'.join(valid_n))
open("KICH_normal_test.txt", 'w').write('\n'.join(test_n))


n1 = [len(glob.glob("/ssd_scratch/cvit/medicalImaging/PATCHES_KICH/cancer/{}/*.png".format(f))) for f in train_c]
n2 = [len(glob.glob("/ssd_scratch/cvit/medicalImaging/PATCHES_KICH/cancer/{}/*.png".format(f))) for f in valid_c]
n3 = [len(glob.glob("/ssd_scratch/cvit/medicalImaging/PATCHES_KICH/cancer/{}/*.png".format(f))) for f in test_c]
n4 = [len(glob.glob("/ssd_scratch/cvit/medicalImaging/PATCHES_KICH/normal/{}/*.png".format(f))) for f in train_n]
n5 = [len(glob.glob("/ssd_scratch/cvit/medicalImaging/PATCHES_KICH/normal/{}/*.png".format(f))) for f in valid_n]
n6 = [len(glob.glob("/ssd_scratch/cvit/medicalImaging/PATCHES_KICH/normal/{}/*.png".format(f))) for f in test_n]

cancer_stats = [(len(i), sum(i)) for i in [n1, n2, n3]]
normal_stats = [(len(i), sum(i)) for i in [n4, n5, n6]]

print("Cancer Stats", cancer_stats)
print("Normal Stats", normal_stats)
