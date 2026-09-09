"""Unchanged original decoder method used only by CPU parity tests."""
import torch
from types import SimpleNamespace as DeliberativeCarrierGeneration
from typing import Sequence,Mapping
THINKING_OPEN="<think>\n"

class CapturedGenerationError(RuntimeError):
    def __init__(self,message,reasoning_text,token_ids):
        super().__init__(message)
        self.reasoning_text=reasoning_text
        self.token_ids=list(token_ids)

class NativeReference:
    @torch.no_grad()
    def generate(
        self,
        *,
        carrier: torch.Tensor,
        messages: Sequence[Mapping[str, str]],
        reasoning_max_new_tokens: int,
        vector_length: int = 3,
    ) -> DeliberativeCarrierGeneration:
        """Run native greedy Receiver reasoning, then constrained I3 readout."""

        if reasoning_max_new_tokens <= 0:
            raise ValueError("reasoning_max_new_tokens must be positive")
        if vector_length not in (3, 6, 9):
            raise ValueError("vector_length must be 3, 6, or 9")
        prompt = self._prompt_embeddings(messages, carrier)
        positions = torch.arange(
            int(prompt.shape[1]), dtype=torch.long, device=prompt.device
        )
        output = self.model(
            inputs_embeds=prompt,
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        next_logits = output.logits[:, -1, :]
        next_position = int(prompt.shape[1])
        reasoning: list[int] = []
        stop_reason = "max_new_tokens"
        for _ in range(reasoning_max_new_tokens):
            next_id = int(torch.argmax(next_logits, dim=-1).item())
            if next_id in self.eos_ids:
                stop_reason = "eos_mapped_to_thinking_close"
                break
            reasoning.append(next_id)
            token = torch.tensor([[next_id]], dtype=torch.long, device=prompt.device)
            position = torch.tensor([next_position], dtype=torch.long, device=prompt.device)
            output = self.model(
                input_ids=token,
                position_ids=position.unsqueeze(0),
                cache_position=position,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values
            next_logits = output.logits[:, -1, :]
            next_position += 1
            if tuple(reasoning[-len(self.close_ids) :]) == self.close_ids:
                stop_reason = "thinking_close"
                break
        if stop_reason == "max_new_tokens":
            raise CapturedGenerationError("deliberative carrier reasoning did not emit </think> within the cap", THINKING_OPEN + self.tokenizer.decode(reasoning, skip_special_tokens=False, clean_up_tokenization_spaces=False), reasoning)
        if stop_reason == "eos_mapped_to_thinking_close":
            for forced_id in self.close_ids:
                reasoning.append(forced_id)
                token = torch.tensor(
                    [[forced_id]], dtype=torch.long, device=prompt.device
                )
                position = torch.tensor(
                    [next_position], dtype=torch.long, device=prompt.device
                )
                output = self.model(
                    input_ids=token,
                    position_ids=position.unsqueeze(0),
                    cache_position=position,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = output.past_key_values
                next_logits = output.logits[:, -1, :]
                next_position += 1

        for forced_id in self.answer_prefix_ids:
            token = torch.tensor([[forced_id]], dtype=torch.long, device=prompt.device)
            position = torch.tensor([next_position], dtype=torch.long, device=prompt.device)
            output = self.model(
                input_ids=token,
                position_ids=position.unsqueeze(0),
                cache_position=position,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values
            next_logits = output.logits[:, -1, :]
            next_position += 1

        vector: list[str] = []
        scores: list[tuple[float, float]] = []
        for offset in range(vector_length):
            logprobs = torch.log_softmax(next_logits.float(), dim=-1)[0]
            restricted = tuple(float(logprobs[value].item()) for value in self.digit_ids)
            bit = min(("0", "1"), key=lambda value: (-restricted[int(value)], value))
            vector.append(bit)
            scores.append(restricted)
            forced = (self.digit_ids[int(bit)],) + (
                self.separator_ids if offset + 1 < vector_length else ()
            )
            for forced_id in forced:
                token = torch.tensor(
                    [[forced_id]], dtype=torch.long, device=prompt.device
                )
                position = torch.tensor(
                    [next_position], dtype=torch.long, device=prompt.device
                )
                output = self.model(
                    input_ids=token,
                    position_ids=position.unsqueeze(0),
                    cache_position=position,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = output.past_key_values
                next_logits = output.logits[:, -1, :]
                next_position += 1
        reasoning_text = THINKING_OPEN + self.tokenizer.decode(
            reasoning,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return DeliberativeCarrierGeneration(
            vector="".join(vector),
            bit_logprobs=tuple(scores),
            reasoning_text=reasoning_text,
            reasoning_token_ids=tuple(reasoning),
            reasoning_token_count=len(reasoning),
            reasoning_stop_reason=stop_reason,
            reasoning_cap_hit=False,
        )
