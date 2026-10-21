"""Dataset identities; VTAB class counts follow the local GPS ViT scripts."""

VTAB_GROUPS = {
    "natural": ("caltech101", "cifar100", "dtd", "flowers102", "pets", "sun397", "svhn"),
    "specialized": ("patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy"),
    "structured": ("clevr_count", "clevr_dist", "dmlab", "kitti", "dsprites_loc",
                   "dsprites_ori", "smallnorb_azi", "smallnorb_ele"),
}
VTAB_CLASSES = dict(caltech101=102, cifar100=100, dtd=47, flowers102=102, pets=37,
                    sun397=397, svhn=10, patch_camelyon=2, eurosat=10, resisc45=45,
                    diabetic_retinopathy=5, clevr_count=8, clevr_dist=6, dmlab=6,
                    kitti=4, dsprites_loc=16, dsprites_ori=16, smallnorb_azi=18,
                    smallnorb_ele=9)

# The uploaded GPS/VTAB bundle uses the original TensorFlow dataset directory
# names for these three tasks.  Keep logical experiment names stable while
# resolving their on-disk locations explicitly.
VTAB_DIRECTORIES = dict(cifar100="cifar", flowers102="oxford_flowers102",
                        pets="oxford_iiit_pet")
