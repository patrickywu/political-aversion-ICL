import openai
from openai import OpenAI
import pandas as pd
import time
from tqdm import tqdm
from joblib import delayed, Parallel
from sentence_transformers import SentenceTransformer
import faiss

# NOTE: this assumes that you will use the OpenAI client. The code needs to be modified if you are using another API.

class gen_llm_clf():
    def __init__(self,
                 df,
                 df_text_feature,
                 df_tweet_id_feature,
                 df_label_feature,
                 openai_client,
                 def_prompt=None,
                 ex_prompt=None,
                 clf_prompt=None,
                 extraction_prompt=None,
                 database=None,
                 k=None,
                 concept_contains=None):

        # housecleaning
        self.df = df
        self.df_text_feature = df_text_feature
        self.df_tweet_id_feature = df_tweet_id_feature
        self.df_label_feature = df_label_feature
        self.def_prompt = def_prompt
        self.ex_prompt = ex_prompt
        self.clf_prompt = clf_prompt

        # Get text and labels
        self.text = self.df[self.df_text_feature].tolist()
        self.tweet_id = self.df[self.df_tweet_id_feature].tolist()
        self.label = self.df[self.df_label_feature].tolist()

        # Get text and labels from the database
        self.database = database
        self.k = k
        self.concept_contains = concept_contains

        # Get the embeddings for the database
        if self.database is not None:
            self.db_texts = self.database[self.df_text_feature].tolist()
            self.db_labels = self.database[self.df_label_feature].tolist()

            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

            self.db_embeddings = self.embedding_model.encode(self.db_texts, convert_to_tensor=False)
            self.index = faiss.IndexFlatL2(self.db_embeddings.shape[1])
            self.index.add(self.db_embeddings)

        # set extraction prompts
        self.extraction_prompt = extraction_prompt

        # get prompts and extraction_prompts
        self.prompts = self.create_prompts()
        self.extraction_prompts = self.create_extraction_prompts()

        # initiate the OpenAI client
        self.client = openai_client

    def retriever(self, query):
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        distances, indices = self.index.search(query_embedding, self.k)

        ret_texts, ret_labels = [self.db_texts[idx] for idx in indices[0]], [self.db_labels[idx] for idx in indices[0]]

        return self.create_context(ret_texts, ret_labels)

    def create_context(self, text, labels):
        context = ''

        for i in range(len(text)):
            if labels[i]==1:
                label_word = 'Yes'
            else:
                label_word = 'No'

            context = context + str(i+1) + '. Example Tweet: "' + text[i] + '"\n Above Example Tweet expresses ' + self.concept_contains + ': ' + label_word + '\n'

        return context

    def create_prompts(self):
        prompts = []
        for i in range(len(self.text)):
            if self.def_prompt is None and self.database is None:
                p = 'INSTRUCTIONS: Answer the QUESTION about the TWEET.' +\
                '\n\nQUESTION: ' + self.clf_prompt +\
                '\n\nTWEET: "' + self.text[i] + '"'
            elif self.def_prompt is not None and self.database is None:
                p = 'INSTRUCTIONS: Answer the QUESTION about the TWEET using the DEFINITION below.' +\
                '\n\nDEFINITION: ' + self.def_prompt +\
                '\n\nQUESTION: Considering the DEFINITION above, ' + self.clf_prompt +\
                '\n\nTWEET: "' + self.text[i] + '"'
            elif self.def_prompt is None and self.database is not None:
                p = 'INSTRUCTIONS: Answer the QUESTION about the TWEET using the EXAMPLE TWEETS below.' +\
                '\n\nEXAMPLE TWEETS: ' + self.ex_prompt + '\n' + self.retriever(self.text[i]) +\
                '\nQUESTION: Considering the EXAMPLE TWEETS above, ' + self.clf_prompt +\
                '\n\nTWEET: "' + self.text[i] + '"'
            else:
                p = 'INSTRUCTIONS: Answer the QUESTION about the TWEET using the DEFINITION and EXAMPLE TWEETS below.' +\
                '\n\nDEFINITION: ' + self.def_prompt +\
                '\n\nEXAMPLE TWEETS: ' + self.ex_prompt + '\n' + self.retriever(self.text[i]) +\
                '\nQUESTION: Considering the DEFINITION and EXAMPLE TWEETS above, ' + self.clf_prompt +\
                '\n\nTWEET: "' + self.text[i] + '"'
            prompts.append(p)
        return prompts

    def create_extraction_prompts(self):
        extraction_prompts = []
        for i in range(len(self.text)):
            extraction_prompts.append(self.extraction_prompt)
        return extraction_prompts

    def prompting_openai(self,
                         prompt,
                         extraction_prompt,
                         system_prompt=None,
                         model='gpt-4o-mini',
                         temp=1.0,
                         top_p=1.0):

        sleepy_times = [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256]

        if system_prompt is not None:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
                ]
        else:
            messages = [
                {'role': 'user', 'content': prompt}
                ]

        for i in range(len(sleepy_times)):
            try:
                completion = self.client.chat.completions.create(model=model,
                                                                 messages=messages,
                                                                 temperature=temp,
                                                                 top_p=top_p)
                reasoning = completion.choices[0].message.content
                break
            except Exception as e:
                print(f"Error: {e}")
                print('Sleep: ' + str(sleepy_times[i]))
                time.sleep(sleepy_times[i])

        messages.append({'role': 'assistant', 'content': reasoning})

        messages.append({'role': 'user', 'content': extraction_prompt})

        # extraction stage
        for j in range(len(sleepy_times)):
            try:
                completion = self.client.chat.completions.create(model=model,
                                                                 messages=messages,
                                                                 temperature=0.0)
                response = completion.choices[0].message.content
                break
            except Exception as e:
                print(f"Error: {e}")
                print('Sleep: ' + str(sleepy_times[i]))
                time.sleep(sleepy_times[i])

        return (reasoning, response)

    def parallelize_openai_calls(self,
                                 prompts,
                                 extraction_prompts,
                                 chunk_size,
                                 system_prompt=None,
                                 model='gpt-4o-mini',
                                 temp=1.0,
                                 top_p=1.0):

        results_collection = []
        for data_chunk in tqdm(self.chunked_data(prompts, extraction_prompts, chunk_size), total=len(prompts)//chunk_size):
            results = Parallel(n_jobs=chunk_size, backend='threading')(delayed(self.prompting_openai)(p, q, system_prompt, model, temp, top_p) for p,q in zip(data_chunk[0], data_chunk[1]))
            results_collection.extend(results)
        return results_collection

    # Yield successive chunk_size chunks from data.
    def chunked_data(self, data, extraction_prompts, chunk_size):
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size], extraction_prompts[i:i+chunk_size]

    # this function simply removes the period at the sentences
    def lowercase_and_remove_period(self, sentence):
        sentence = sentence.lower()

        if sentence.endswith('.'):
            sentence = sentence[:-1]

        return sentence

    def check_extraction_results(self, extraction_results):
        counter = 0
        for i in range(len(extraction_results)):
            if extraction_results[i] != 'yes' and extraction_results[i] != 'no':
                print(i)
                print(extraction_results[i])
                counter = counter + 1

        if counter==0:
            return None
        else:
            return counter

    def make_final_df(self,
                      clf_reasoning,
                      extraction_results):

        binary_extraction_results = [1 if x=='yes' else 0 for x in extraction_results]

        classification_results = pd.DataFrame({'tweet_id': self.tweet_id,
                                               'text': self.text,
                                               'label': self.label,
                                               'llm_response': clf_reasoning,
                                               'final_clf': binary_extraction_results})

        return classification_results

class gen_llm_clf_multilabel():
    def __init__(self,
                 df,
                 df_text_feature,
                 df_tweet_id_feature,
                 df_label_feature,
                 df_label_text_desc,
                 openai_client,
                 def_prompt=None,
                 ex_prompt=None,
                 clf_prompt=None,
                 extraction_prompt=None,
                 database=None,
                 k=None,
                 concept_contains=None):

        # housecleaning
        self.df = df
        self.df_text_feature = df_text_feature
        self.df_tweet_id_feature = df_tweet_id_feature
        self.df_label_feature = df_label_feature
        self.df_label_text_desc = df_label_text_desc
        self.def_prompt = def_prompt
        self.ex_prompt = ex_prompt
        self.clf_prompt = clf_prompt

        # Get text and labels
        self.text = self.df[self.df_text_feature].tolist()
        self.tweet_id = self.df[self.df_tweet_id_feature].tolist()
        self.label = self.df[self.df_label_feature].tolist()

        # Get text and labels from the database
        self.database = database
        self.k = k
        self.concept_contains = concept_contains

        # Get the embeddings for the database
        if self.database is not None:
            self.db_texts = self.database[self.df_text_feature].tolist()
            self.db_labels = self.database[self.df_label_feature].tolist()

            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

            self.db_embeddings = self.embedding_model.encode(self.db_texts, convert_to_tensor=False)
            self.index = faiss.IndexFlatL2(self.db_embeddings.shape[1])
            self.index.add(self.db_embeddings)

        # set extraction prompts
        self.extraction_prompt = extraction_prompt

        # get prompts and extraction_prompts
        self.prompts = self.create_prompts()
        self.extraction_prompts = self.create_extraction_prompts()

        # initiate the OpenAI client
        self.client = openai_client

    def retriever(self, query):
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        distances, indices = self.index.search(query_embedding, self.k)

        ret_texts, ret_labels = [self.db_texts[idx] for idx in indices[0]], [self.db_labels[idx] for idx in indices[0]]

        return self.create_context(ret_texts, ret_labels, self.df_label_text_desc)

    def create_context(self, text, labels, labels_text):
        context = ''
        for i in range(len(text)):
            label_word = labels_text[labels[i]]
            context = context + str(i+1) + '. Example Tweet: "' + text[i] + '"\n Above Example Tweet expresses ' + self.concept_contains + ': ' + label_word + '\n'

        return context

    def create_prompts(self):
        prompts = []
        for i in range(len(self.text)):
            if self.def_prompt is None and self.database is None:
                p = 'INSTRUCTIONS: Answer the QUESTION about the TWEET.' +\
                '\n\nQUESTION: ' + self.clf_prompt +\
                '\n\nTWEET: "' + self.text[i] + '"'
            elif self.def_prompt is not None and self.database is None:
                p = 'INSTRUCTIONS: Answer the QUESTION about the TWEET using the DEFINITION below.' +\
                '\n\nDEFINITION: ' + self.def_prompt +\
                '\n\nQUESTION: Considering the DEFINITION above, ' + self.clf_prompt +\
                '\n\nTWEET: "' + self.text[i] + '"'
            elif self.def_prompt is None and self.database is not None:
                p = 'INSTRUCTIONS: Answer the QUESTION about the TWEET using the EXAMPLE TWEETS below.' +\
                '\n\nEXAMPLE TWEETS: ' + self.ex_prompt + '\n' + self.retriever(self.text[i]) +\
                '\nQUESTION: Considering the EXAMPLE TWEETS above, ' + self.clf_prompt +\
                '\n\nTWEET: "' + self.text[i] + '"'
            else:
                p = 'INSTRUCTIONS: Answer the QUESTION about the TWEET using the DEFINITION and EXAMPLE TWEETS below.' +\
                '\n\nDEFINITION: ' + self.def_prompt +\
                '\n\nEXAMPLE TWEETS: ' + self.ex_prompt + '\n' + self.retriever(self.text[i]) +\
                '\nQUESTION: Considering the DEFINITION and EXAMPLE TWEETS above, ' + self.clf_prompt +\
                '\n\nTWEET: "' + self.text[i] + '"'
            prompts.append(p)
        return prompts

    def create_extraction_prompts(self):
        extraction_prompts = []
        for i in range(len(self.text)):
            extraction_prompts.append(self.extraction_prompt)
        return extraction_prompts

    def prompting_openai(self,
                         prompt,
                         extraction_prompt,
                         system_prompt=None,
                         model='gpt-4o-mini',
                         temp=1.0,
                         top_p=1.0):

        sleepy_times = [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256]

        if system_prompt is not None:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
                ]
        else:
            messages = [
                {'role': 'user', 'content': prompt}
                ]

        for i in range(len(sleepy_times)):
            try:
                completion = self.client.chat.completions.create(model=model,
                                                                 messages=messages,
                                                                 temperature=temp,
                                                                 top_p=top_p)
                reasoning = completion.choices[0].message.content
                break
            except Exception as e:
                print(f"Error: {e}")
                print('Sleep: ' + str(sleepy_times[i]))
                time.sleep(sleepy_times[i])

        messages.append({'role': 'assistant', 'content': reasoning})

        messages.append({'role': 'user', 'content': extraction_prompt})

        # extraction stage
        for j in range(len(sleepy_times)):
            try:
                completion = self.client.chat.completions.create(model=model,
                                                                 messages=messages,
                                                                 temperature=0.0)
                response = completion.choices[0].message.content
                break
            except Exception as e:
                print(f"Error: {e}")
                print('Sleep: ' + str(sleepy_times[i]))
                time.sleep(sleepy_times[i])

        return (reasoning, response)

    def parallelize_openai_calls(self,
                                 prompts,
                                 extraction_prompts,
                                 chunk_size,
                                 system_prompt=None,
                                 model='gpt-4o-mini',
                                 temp=1.0,
                                 top_p=1.0):

        results_collection = []
        for data_chunk in tqdm(self.chunked_data(prompts, extraction_prompts, chunk_size), total=len(prompts)//chunk_size):
                results = Parallel(n_jobs=chunk_size, backend='threading')(delayed(self.prompting_openai)(p, q, system_prompt, model, temp, top_p) for p,q in zip(data_chunk[0], data_chunk[1]))
                results_collection.extend(results)
        return results_collection

    # Yield successive chunk_size chunks from data.
    def chunked_data(self, data, extraction_prompts, chunk_size):
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size], extraction_prompts[i:i+chunk_size]

    # this function simply removes the period at the sentences
    def lowercase_and_remove_period(self, sentence):
        sentence = sentence.lower()

        if sentence.endswith('.'):
            sentence = sentence[:-1]

        return sentence

    def check_extraction_results(self, extraction_results):
        counter = 0
        for i in range(len(extraction_results)):
            if extraction_results[i] not in self.df_label_text_desc:
                print(i)
                print(extraction_results[i])
                counter = counter + 1

        if counter==0:
            return None
        else:
            return counter

    def make_final_df(self,
                      clf_reasoning,
                      extraction_results):

        numeric_extraction_results = [self.df_label_text_desc.index(x) for x in extraction_results]

        classification_results = pd.DataFrame({'tweet_id': self.tweet_id,
                                               'text': self.text,
                                               'label': self.label,
                                               'llm_response': clf_reasoning,
                                               'final_clf': numeric_extraction_results})

        return classification_results