import copy
import gc
import logging

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed

from nanogcg.utils import INIT_CHARS, find_executable_batch_size, get_nonascii_toks, mellowmax

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"

@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]

class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = [] # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
            return

        self.buffer[-1] = (loss, optim_ids)
        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]
    
    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]
    
    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)

def sample_ids_from_grad(
    ids: Tensor, 
    grad: Tensor, 
    search_width: int, 
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    ### Note that this is different than GCG; in AutoPrompt, we choose a fix random pos instead of best ones
    sampled_ids_pos = torch.randint(0, n_optim_tokens, (1,), device=grad.device)
    sampled_ids_pos = sampled_ids_pos.repeat(search_width, n_replace)

    # sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids

def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
           filtered_ids.append(ids[i]) 
    
    if not filtered_ids:
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )
    
    return torch.stack(filtered_ids)

class GCG:
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)
        self.prefix_cache = None

        self.stop_flag = False

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")
    
    def run(
        self,
        messages_list: List[Union[str, List[dict]]],
        targets: List[str],
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
    
        self.messages_list = [self._prepare_messages(messages) for messages in messages_list]
        self.targets = targets

        self._prepare_inputs()

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        
        for _ in tqdm(range(config.num_steps)):
            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids) 

            with torch.no_grad():
                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences 
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                # loss = self.compute_candidates_loss(batch_size, sampled_ids)
                loss = find_executable_batch_size(self.compute_candidates_loss, batch_size)(sampled_ids)

                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)                

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.") 
                break
              
        min_loss_index = losses.index(min(losses)) 

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )

        return result
    
    def _prepare_messages(self, messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)
    
        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"
        
        return messages

    def _prepare_inputs(self):
        tokenizer = self.tokenizer
        config = self.config

        self.before_ids_list = []
        self.after_ids_list = []
        self.target_ids_list = []

        for messages, target in zip(self.messages_list, self.targets):
            template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
            # Remove the BOS token -- this will get added when tokenizing, if necessary
            if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
                template = template.replace(tokenizer.bos_token, "")
            before_str, after_str = template.split("{optim_str}")

            target = " " + target if config.add_space_before_target else target

            # Tokenize everything that doesn't get optimized
            before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(self.model.device).to(torch.int64)
            after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model.device).to(torch.int64)
            target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model.device).to(torch.int64)

            self.before_ids_list.append(before_ids)
            self.after_ids_list.append(after_ids)
            self.target_ids_list.append(target_ids)

        # Embed everything that doesn't get optimized
        self.before_embeds_list = [self.embedding_layer(ids) for ids in self.before_ids_list]
        self.after_embeds_list = [self.embedding_layer(ids) for ids in self.after_ids_list]
        self.target_embeds_list = [self.embedding_layer(ids) for ids in self.target_ids_list]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            self.prefix_cache_list = []
            with torch.no_grad():
                for before_embeds in self.before_embeds_list:
                    output = self.model(inputs_embeds=before_embeds, use_cache=True)
                    self.prefix_cache_list.append(output.past_key_values)
    
    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids
                
        else: # assume list
            if (len(config.optim_str_init) != config.buffer_size):
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size) 

        # Compute the loss on the initial buffer entries
        init_buffer_losses = self.compute_candidates_loss(true_buffer_size, init_buffer_ids)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        logger.info("Initialized attack buffer.")
        
        return buffer
    
    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(dtype=model.dtype, device=model.device)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        total_loss = 0
        for i in range(len(self.messages_list)):
            if self.config.use_prefix_cache:
                input_embeds = torch.cat([optim_embeds, self.after_embeds_list[i], self.target_embeds_list[i]], dim=1)
                output = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache_list[i])
            else:
                input_embeds = torch.cat([self.before_embeds_list[i], optim_embeds, self.after_embeds_list[i], self.target_embeds_list[i]], dim=1)
                output = model(inputs_embeds=input_embeds)

            logits = output.logits

            # Shift logits so token n-1 predicts token n
            shift = input_embeds.shape[1] - self.target_ids_list[i].shape[1]
            shift_logits = logits[..., shift-1:-1, :].contiguous() # (1, num_target_ids, vocab_size)
            shift_labels = self.target_ids_list[i]

            if self.config.use_mellowmax:
                label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
            else:
                loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            total_loss += loss

        total_loss /= len(self.messages_list)
        optim_ids_onehot_grad = torch.autograd.grad(outputs=[total_loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad
    
    def compute_candidates_loss(
        self,
        search_batch_size: int, 
        sampled_ids: Tensor, 
    ) -> Tensor:
        all_loss = []

        for i in range(0, sampled_ids.shape[0], search_batch_size):
            with torch.no_grad():
                sampled_ids_batch = sampled_ids[i:i+search_batch_size]
                current_batch_size = sampled_ids_batch.shape[0]

                batch_loss = 0
                for j in range(len(self.messages_list)):
                    if self.config.use_prefix_cache:
                        prefix_cache_batch = [
                            [x.expand(current_batch_size, -1, -1, -1) for x in layer]
                            for layer in self.prefix_cache_list[j]
                        ]

                        input_embeds = torch.cat([
                            self.embedding_layer(sampled_ids_batch),
                            self.after_embeds_list[j].repeat(current_batch_size, 1, 1),
                            self.target_embeds_list[j].repeat(current_batch_size, 1, 1),
                        ], dim=1)
                        outputs = self.model(inputs_embeds=input_embeds, past_key_values=prefix_cache_batch)
                    else:
                        input_embeds = torch.cat([
                            self.before_embeds_list[j].repeat(current_batch_size, 1, 1),
                            self.embedding_layer(sampled_ids_batch),
                            self.after_embeds_list[j].repeat(current_batch_size, 1, 1),
                            self.target_embeds_list[j].repeat(current_batch_size, 1, 1),
                        ], dim=1)
                        outputs = self.model(inputs_embeds=input_embeds)

                    logits = outputs.logits

                    tmp = input_embeds.shape[1] - self.target_ids_list[j].shape[1]
                    shift_logits = logits[..., tmp-1:-1, :].contiguous()
                    shift_labels = self.target_ids_list[j].repeat(current_batch_size, 1)
                    
                    if self.config.use_mellowmax:
                        label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                        loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                    else:
                        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")

                    loss = loss.view(current_batch_size, -1).mean(dim=-1)
                    batch_loss += loss

                    if self.config.early_stop:
                        if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                            self.stop_flag = True

                batch_loss /= len(self.messages_list)
                all_loss.append(batch_loss)

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

# A wrapper around the GCG `run` method that provides a simple API
def run_multi_autoprompt(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages_list: List[Union[str, List[dict]]],
    targets: List[str],
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    """Generates a single optimized string using GCG for multiple prompts simultaneously. 

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages_list: A list of conversations to use for optimization.
        targets: A list of target generations, one for each conversation.
        config: The GCG configuration to use.
    
    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()
    
    logger.setLevel(getattr(logging, config.verbosity))
    
    gcg = GCG(model, tokenizer, config)
    result = gcg.run(messages_list, targets)
    return result