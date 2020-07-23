import shutil
import fnmatch
import os

for file in os.listdir('C:\Users\Lenovo\Documents\NTU course materials\EEE\EE4208 INTELLIGENT SYSTEMS DESIGN\caspeal\FRONTAL\Expression'):
    if fnmatch.fnmatch(file, '[EC]'):
        newPath = shutil.copy(file, 'C:\Users\Lenovo\Documents\NTU course materials\EEE\EE4208 INTELLIGENT SYSTEMS DESIGN\caspeal\FRONTAL\Expression\EC')
    if fnmatch.fnmatch(file, '[EF]'):
        newPath = shutil.copy(file, 'C:\Users\Lenovo\Documents\NTU course materials\EEE\EE4208 INTELLIGENT SYSTEMS DESIGN\caspeal\FRONTAL\Expression\EF')
    if fnmatch.fnmatch(file, '[EL]'):
        newPath = shutil.copy(file, 'C:\Users\Lenovo\Documents\NTU course materials\EEE\EE4208 INTELLIGENT SYSTEMS DESIGN\caspeal\FRONTAL\Expression\EL')
    if fnmatch.fnmatch(file, '[EO]'):
        newPath = shutil.copy(file, 'C:\Users\Lenovo\Documents\NTU course materials\EEE\EE4208 INTELLIGENT SYSTEMS DESIGN\caspeal\FRONTAL\Expression\EO')
    if fnmatch.fnmatch(file, '[ES]'):
        newPath = shutil.copy(file, 'C:\Users\Lenovo\Documents\NTU course materials\EEE\EE4208 INTELLIGENT SYSTEMS DESIGN\caspeal\FRONTAL\Expression\ES')