from transformers import AutoTokenizer


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


class MultiTurnLossMaskGenerator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        tokenizer_type: str = "qwen",
        chat_template_kwargs: dict | None = None,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.system_message_length, self.gen_token_length = self.get_system_message_length()

    def _act(self, messages, *, tokenize: bool, add_generation_prompt: bool = False, add_special_tokens: bool = False):
        """
        Small helper so all apply_chat_template calls share the same kwargs.
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            add_special_tokens=add_special_tokens,
            **self.chat_template_kwargs,
        )

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        return get_response_lengths(loss_masks)

    def find_all_sublist_indices(self, main_list, sublist):
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def get_system_message_length(self) -> tuple[int, int]:
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]

        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]

        chat_template_text = self._act(
            test_messages,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        chat_template_token_ids = self.tokenizer(chat_template_text, add_special_tokens=False)["input_ids"]

        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2

        gen_ids = self._act(
            test_messages,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        gen_token_length = len(gen_ids) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def gen_multi_turn_loss_mask_qwen(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        for i, message in enumerate(messages):
            message_ids = self._act([message], tokenize=True, add_special_tokens=False)

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        prefix_message = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}
        prefix_token_ids = self._act([prefix_message], tokenize=True, add_special_tokens=False)

        for i, message in enumerate(messages):
            prefixed_message_ids = self._act([prefix_message, message], tokenize=True, add_special_tokens=False)
            message_ids = prefixed_message_ids[len(prefix_token_ids) :]

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_distill_qwen(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        prompt = self._act(messages[:1], tokenize=False, add_generation_prompt=True, add_special_tokens=False)
        response = messages[-1]["content"]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        response_length = len(response_tokens)
        token_ids = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * response_length

        if messages[-1].get("step_loss_mask", 1) != 1:
            loss_mask = [0] * len(token_ids)
        return token_ids, loss_mask

    def get_loss_mask(self, messages: list[dict]) -> list[int]:
        if self.tokenizer_type == "qwen":
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                return self.gen_multi_turn_loss_mask_distill_qwen(messages)

            return self.gen_multi_turn_loss_mask_qwen(messages)
        elif self.tokenizer_type == "qwen3":
            return self.gen_multi_turn_loss_mask_qwen3(messages)
        elif self.tokenizer_type == "distill_qwen":
            return self.gen_multi_turn_loss_mask_distill_qwen(messages)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def get_text_from_loss_mask(self, token_ids: list[int], loss_masks: list[int]) -> list[str]:
        selected_texts = []
        current_tokens = []

        for idx, mask in enumerate(loss_masks):
            if mask == 1:
                current_tokens.append(token_ids[idx])
            elif current_tokens:
                selected_texts.append(self.tokenizer.decode(current_tokens))
                current_tokens = []

        if current_tokens:
            selected_texts.append(self.tokenizer.decode(current_tokens))

        return selected_texts
