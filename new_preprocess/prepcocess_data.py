import json
import os
import sys
import spacy
import wikipediaapi
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
from constants import TEAM_INFO_KEYS, PLAYER_INFO_KEYS
from table_utils import parse_table_to_text


class Summarizer:
    def __init__(self, batch_size=8, use_gpu=True):
        self.model_name = 'sshleifer/distilbart-cnn-12-6'
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)

        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.model.to(self.device)
        self.batch_size = batch_size

    def summarize_batch(self, texts, max_length=1024, summary_length=100):
        summaries = []
        texts = [" ".join(text) if isinstance(text, list) else text for text in texts]

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Summarizing batches"):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(self.device)
            
            summary_ids = self.model.generate(
                inputs["input_ids"], 
                max_length=summary_length, 
                min_length=40, 
                num_beams=4, 
                early_stopping=True
            )

            batch_summaries = [self.tokenizer.decode(s, skip_special_tokens=True) for s in summary_ids]
            summaries.extend(batch_summaries)

        return summaries


class EntityResolver:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.wiki = wikipediaapi.Wikipedia('en')
        self.cache = {} 

    def get_wikipedia_entity(self, entity_name):
        if entity_name in self.cache:
            return self.cache[entity_name]

        try:
            page = self.wiki.page(entity_name)
            if not page.exists():
                return "No knowledge found."
            description = page.summary.split(".")[0] + "."  
            self.cache[entity_name] = description
            return description
        except Exception:
            return "No knowledge found."

    def resolve_entities(self, texts):
        resolved_texts = []

        docs = list(self.nlp.pipe(texts, batch_size=16))

        for doc, text in tqdm(zip(docs, texts)):
            resolved_entities = {ent.text: self.get_wikipedia_entity(ent.text) for ent in doc.ents}

            for entity, description in resolved_entities.items():
                if description != "No knowledge found." and entity not in text:
                    text = text.replace(entity, f"{entity} ({description})")

            resolved_texts.append(text)

        return resolved_texts


def split_filtered_relations(relations):
    team_relations = set()
    player_relations = set()
    for _, num, rel, label in relations:
        if isinstance(label, bool):
            team_relations.add((num[3], rel, label))
        elif isinstance(label, str):
            player_relations.add((num[3], rel, label))
        else:
            assert label is None
    return list(team_relations), list(player_relations)


def get_filtered_team_table(original, team_relations):
    home_relations = {rel: num for num, rel, label in team_relations if label}
    vis_relations = {rel: num for num, rel, label in team_relations if not label}

    keyset = set(home_relations.keys()).union(set(vis_relations.keys()))
    keys = [key for key in TEAM_INFO_KEYS if key in keyset]

    data = [[""], [original["home_line"]["TEAM-NAME"]], [original['vis_line']['TEAM-NAME']]]
    for k in keys:
        data[0].append(TEAM_INFO_KEYS[k])
        data[1].append(home_relations.get(k, ""))
        data[2].append(vis_relations.get(k, ""))

    if not vis_relations:
        data.pop(2)
    if not home_relations:
        data.pop(1)
    if len(data) == 1:
        return []
    return data


def get_filtered_player_table(original, player_relations):
    player_ids = sorted(set(label for _, _, label in player_relations), key=int)
    per_player_relations = {pid: {"PLAYER_NAME": original['box_score']['PLAYER_NAME'][pid]} for pid in player_ids}

    for num, rel, label in player_relations:
        assert rel.startswith("PLAYER-")
        per_player_relations[label][rel[len("PLAYER-"):]] = num

    keyset = {k for relations in per_player_relations.values() for k in relations}
    keys = [key for key in PLAYER_INFO_KEYS if key in keyset]

    return [[PLAYER_INFO_KEYS[k] for k in keys]] + [[per_player_relations[pid].get(k, '') for k in keys] for pid in player_ids]


if __name__ == "__main__":
    _, inp_dir, oup_dir = sys.argv

    summarizer = Summarizer(batch_size=16, use_gpu=True)
    entity_resolver = EntityResolver()

    for split in ['train', 'valid', 'test']:
        with open(os.path.join(inp_dir, f'{split}.json')) as f:
            original = json.load(f)
        with open(os.path.join(inp_dir, f'{split}.relations.json')) as f:
            relations = json.load(f)

        texts = [o.get("summary", "") for o in original]
        texts = [" ".join(text) if isinstance(text, list) else text for text in texts]
       
        enriched_texts = entity_resolver.resolve_entities(texts)
        summarized_texts = summarizer.summarize_batch(enriched_texts)

 
        with open(os.path.join(oup_dir, f'{split}.data'), 'w') as f:
            for o, r, summarized_text in zip(original, relations, summarized_texts):
                o["summary"] = summarized_text

                tr, pr = split_filtered_relations(r)
                team_table = get_filtered_team_table(o, tr)
                player_table = get_filtered_player_table(o, pr)

                text = "Team:\n{}\nPlayer:\n{}".format(
                    parse_table_to_text(team_table),
                    parse_table_to_text(player_table)
                )
                text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
                f.write(text.replace("\n", " <NEWLINE> ").strip() + "\n")