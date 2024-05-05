import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from vanna.base import VannaBase
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.flask import VannaFlaskApp

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y%m%d %H%M%S",
)


class Hf_sqlcoder(VannaBase):
    def __init__(self, config=None):
        model_name = config.get("model_name", None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=True,
        )

    def system_message(self, message: str) -> any:
        pass

    def user_message(self, message: str) -> any:
        pass

    def assistant_message(self, message: str) -> any:
        pass

    def add_ddl_to_prompt(
        self, prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    prompt = prompt.replace("$SCHEMA_TEMPLATE", f"{ddl}\n\n")

        return prompt

    def get_sql_prompt(
        self,
        prompt_template: str,
        question: str,
        ddl_list: list,
        **kwargs,
    ):

        prompt = prompt_template.replace("$QUESTION_TEMPLATE", f"{question}\n\n")
        prompt = self.add_ddl_to_prompt(prompt, ddl_list, max_tokens=14000)
        ##TO-DO: Add sample queries and documentation according to the prompt guidelines of sql-coder

        return prompt

    def generate_sql(self, question: str, **kwargs) -> str:

        prompt_template = self.config.get("prompt_template", None)

        ddl_list = self.get_related_ddl(question, **kwargs)
        prompt = self.get_sql_prompt(
            prompt_template=prompt_template,
            question=question,
            ddl_list=ddl_list,
            **kwargs,
        )
        self.log(prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(llm_response)
        return llm_response

    def submit_prompt(self, prompt: str, **kwargs) -> str:

        eos_token_id = self.tokenizer.eos_token_id
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=300,
            do_sample=False,
            return_full_text=False,  # added return_full_text parameter to prevent splitting issues with prompt
            num_beams=5,  # do beam search with 5 beams for high quality results
        )
        generated_query = (
            pipe(
                prompt,
                num_return_sequences=1,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
            )[0]["generated_text"]
            .split(";")[0]
            .split("```")[0]
            .strip()
            + ";"
        )
        return generated_query


class MyVanna(ChromaDB_VectorStore, Hf_sqlcoder):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Hf_sqlcoder.__init__(self, config=config)


config = {
    "model_name": "defog/sqlcoder-7b-2",
    "prompt_template": """
    ### Task
    Generate a SQL query to answer [QUESTION] $QUESTION_TEMPLATE [/QUESTION]

    ### Instructions
    If you cannot answer the question with the available database schema, return 'I do not know'
    
    ### Database Schema
    The query will run on a database with the following schema:
    $SCHEMA_TEMPLATE

    ### Answer
    Given the database schema, here is the SQL query that answers [QUESTION] $QUESTION_TEMPLATE [/QUESTION]
    [SQL]
    """,
}

vn = MyVanna(config=config)
vn.connect_to_sqlite("gw_comments.sqlite")
df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
for ddl in df_ddl["sql"].to_list():
    vn.train(ddl=ddl)
app = VannaFlaskApp(vn)
app.run(host="0.0.0.0", port=7080)

# TODO: CHECK SQL-CODER AND VANNA schema, question, and OUTPUTS. ARE THEY SAME? https://github.com/defog-ai/sqlcoder/blob/main/metadata.sql
