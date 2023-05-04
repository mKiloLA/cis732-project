import os
import shutil

# os.makedirs('gan_labels', exist_ok=True)
# for i in range(0, 3000):
#     with open('gan_labels/new_data_{}.txt'.format(i), 'w') as f:
#         f.write('1 0.5 0.5 1 1')


# for i in range(0, 3000):
#     os.remove('{}.txt'.format(i))


# for i in range(0, 3000):
#     os.rename('labels/{}.txt'.format(i), 'labels/new_image_{}.txt'.format(i))


# for i in range(0, 3000):
#     os.rename('test_images/{}.png'.format(i), 'test_images/new_data_{}.png'.format(i))


src = 'Bale80-10-10Split/images/test'
dest = 'BaleDataset/images/test'
src_files = os.listdir(src)
num_files = len(src_files)
i = 1
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest)
        print('Completed: {}/{}'.format(i, num_files))
        i+=1


# src_image = 'Bale80-10-10Split/images/train'
# src_label = 'Bale80-10-10Split/labels/train'

# src_files_image = os.listdir(src_image)
# src_files_image = [os.path.splitext(x)[0] for x in src_files_image]
# src_files_label = os.listdir(src_label)
# src_files_label = [os.path.splitext(x)[0] for x in src_files_label]
# missing = []

# for name in src_files_image:
#     if name not in src_files_label:
#         missing.append(name)
#         print(name)

# for name in missing:
#     os.remove('{}/{}.jpg'.format(src_image, name))
