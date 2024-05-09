import logging
from vanna.hf import Hf_sqlcoder
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.flask import VannaFlaskApp

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y%m%d %H%M%S",
)


class MyVanna(ChromaDB_VectorStore, Hf_sqlcoder):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Hf_sqlcoder.__init__(self, config=config)


config = {"model_name": "defog/sqlcoder-7b-2"}

vn = MyVanna(config=config)
# vn.connect_to_sqlite("sample.sqlite")
# df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
# for ddl in df_ddl["sql"].to_list():
#     vn.train(ddl=ddl)
app = VannaFlaskApp(vn)
app.run()
