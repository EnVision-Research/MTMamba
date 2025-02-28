import os

db_root = '/path/to/'
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]

db_names = {'PASCALContext': 'PASCALContext', 'NYUD_MT': 'NYUDv2', 'Cityscapes': 'cityscapes_all'}
db_paths = {}
for database, db_pa in db_names.items():
    db_paths[database] = os.path.join(db_root, db_pa)