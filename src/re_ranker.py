
import os
import requests
from dotenv import load_dotenv

load_dotenv()




































































##-----------------------------------------------JEV PART( open router)------------------------------##

# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# if not OPENROUTER_API_KEY:
#     raise ValueError("OPENROUTER_API_KEY is not set")


# OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"

# JEV_MODEL = "typesafe/jev-1.13"



# def get_jev_relevance(query, document):
#     """
#     Ask JEV whether a retrieved repository chunk
#     is relevant to the user's query.
#     """

#     file_path = document.metadata.get(
#         "path",
#         "unknown",
#     )

#     content = document.page_content

#     payload = {
#         "model": JEV_MODEL,

#         "state": {
#             "query": query,

#             "candidate": {
#                 "file": file_path,
#                 "content": content,
#             },
#         },

#         "questions": {
#             "relevant": {
#                 "type": "choice",

#                 "instructions": (
#                     "Determine whether this repository "
#                     "code chunk is relevant to answering "
#                     "the user's query."
#                 ),

#                 "criteria": {
#                     "relevant": (
#                         "The code chunk directly contains "
#                         "or explains information needed "
#                         "to answer the user's query."
#                     ),

#                     "not_relevant": (
#                         "The code chunk does not contain "
#                         "useful information for answering "
#                         "the user's query."
#                     ),
#                 },
#             }
#         },
#     }

#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json",
#     }

#     response = requests.post(
#         OPENROUTER_URL,
#         headers=headers,
#         json=payload,
#         timeout=30,
#     )

#     response.raise_for_status()

#     data = response.json()

#     answer = data["answers"]["relevant"]

#     return {
#         "choice": answer["choice"],
#         "confidence": answer["confidence"],
#         "probabilities": answer["probabilities"],
#     }



# def rerank_documents(query, results, top_k=5):
#     """
#     Take the top hybrid candidates and use JEV
#     to determine their relevance to the query.
#     """

#     reranked = []

#     print("\nSending candidates to JEV...")

#     for result in results:

#         document = result["document"]

#         # Ask JEV about this chunk
#         jev_result = get_jev_relevance(
#             query=query,
#             document=document,
#         )

#         # Probability that JEV considers it relevant
#         relevance_score = jev_result[
#             "probabilities"
#         ].get(
#             "relevant",
#             0.0,
#         )

#         # Store JEV information
#         result["jev_score"] = relevance_score
#         result["jev_choice"] = jev_result["choice"]
#         result["jev_confidence"] = jev_result["confidence"]

#         reranked.append(result)

#         print(
#             "-",
#             document.metadata.get("path"),
#             "| chunk:",
#             document.metadata.get("chunk_id"),
#             "| hybrid:",
#             round(
#                 result.get("hybrid_score", 0.0),
#                 4,
#             ),
#             "| JEV:",
#             round(
#                 relevance_score,
#                 4,
#             ),
#             "| confidence:",
#             round(
#                 jev_result["confidence"],
#                 4,
#             ),
#         )

   

#     reranked.sort(
#         key=lambda result: result["jev_score"],
#         reverse=True,
#     )


#     final_results = reranked[:top_k]

#     print(
#         f"\nFinal JEV reranked chunks: "
#         f"{len(final_results)}"
#     )

#     return final_results

