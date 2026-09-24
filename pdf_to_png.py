import pymupdf
from os import listdir
from os.path import isfile, join

# """
dir_path = '/Users/robin/Documents/Master thesis 1/figs/densities_with_const_mass_comp'
onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

# print(onlyfiles)
onlyfiles.sort()
# print(onlyfiles[0:4])

for i,file in enumerate(onlyfiles[1:]):
    if file[-2:-1]=="df":
        doc = pymupdf.open(join(dir_path,file))
        print(file)

        pixmap = doc[0].get_pixmap(dpi=100)
        pixmap.save(join(dir_path,file[:-4]+".png"))

    print(f"{100*(i+1)/len(onlyfiles)}%")
"""
path = '/Users/robin/Documents/Master thesis 1/figs/densities_with_const_mass_comp/Bar_0.5_g_0.2.pdf'
doc = pymupdf.open(path)
pixmap = doc[0].get_pixmap(dpi=100)
pixmap.save(join("/Users/robin/Documents/Master thesis 1/figs/densities_with_const_mass_comp/Bar_0.5_g_0.2.png"))
#"""
