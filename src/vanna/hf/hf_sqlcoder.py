import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from ..base import VannaBase


class Hf_sqlcoder(VannaBase):
    def __init__(self, config=None):
        model_name = self.config.get("model_name", None)
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

        total_ddl = ""

        if len(ddl_list) > 0:
            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(prompt)
                    + self.str_to_approx_token_count(total_ddl)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    total_ddl += f"\n    {ddl}"
        prompt = prompt.replace("$SCHEMA_TEMPLATE", f"{total_ddl}")

        return prompt

    def get_sql_prompt(
        self,
        prompt_template: str,
        question: str,
        ddl_list: list,
        **kwargs,
    ) -> str:

        prompt = prompt_template.replace("$QUESTION_TEMPLATE", f"{question}")
        prompt = self.add_ddl_to_prompt(prompt, ddl_list, max_tokens=14000)

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
        self.log(title="SQL Prompt", message=prompt)
        sql = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=sql)
        return sql

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
        response = (
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
        return response
