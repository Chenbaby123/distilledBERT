from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

# Define the question and context
question, text = "What are the key features of DistilBERT?", "DistilBERT is a smaller version of BERT, designed to be faster and more efficient while retaining most of the performance. It uses a knowledge distillation technique..."
# question = "Who was Jim Henson?"


# Tokenize the inputs
inputs = tokenizer(question, text, return_tensors="pt")

# Get the model's output
with torch.no_grad():
    outputs = model(**inputs)

# Get the start and end logits for the answer
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Convert logits to probabilities and find the highest scoring span
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits)

# Convert token indices back to text
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx+1]))

print("Question:", question)
print("Answer:", answer)

# # 1. Contextual Information Retrieval
# question = "What are the key features of DistilBERT?"
# context = "DistilBERT is a smaller version of BERT, designed to be faster and more efficient while retaining most of the performance. It uses a knowledge distillation technique..."
# # 2. Interactive Chatbots
# question = "What is the mission of Hugging Face?"
# context = "Hugging Face's mission is to democratize good machine learning."
# # 3. Summarization via Extractive QA
# question = "Summarize the key points of this document."
# context = "The document discusses the latest advancements in AI, focusing on language models like BERT, GPT, and their applications..."
# # 4. Semantic Search
# question = "What are the benefits of using Transformers?"
# context = "Transformers are effective because they can model long-range dependencies in text. They also allow parallelization during training..."
# # 5. Document Annotations and Highlighting
# question = "Highlight the challenges mentioned in this document."
# context = "One major challenge in AI is the need for vast amounts of labeled data. Another is the ethical considerations around AI deployment..."
# # 6. Content-Based Filtering and Recommendations
# question = "Recommend articles about ethical AI."
# context = "This article discusses the ethics of AI, focusing on data privacy, bias in algorithms, and the implications for society..."
# # 7. Knowledge Base Question Answering
# question = "Who is the CEO of OpenAI?"
# context = "OpenAI is led by Sam Altman, who has been the CEO since 2019..."
# # 8. FAQ Automation
# question = "How do I reset my password?"
# context = "To reset your password, go to the login page and click on 'Forgot Password.' Follow the instructions sent to your email."
# # 9. Interactive Educational Tools
# question = "What is photosynthesis?"
# context = "Photosynthesis is the process by which green plants use sunlight to synthesize foods with carbon dioxide and water."
# # 10. Legal Document Analysis
# question = "What is the termination clause?"
# context = "In the event of termination, either party must provide 30 days' notice. The termination clause specifies..."
# # 11. Data Extraction from Structured Documents
# question = "What is the total amount?"
# context = "Invoice #12345, Date: 2024-08-09. Total Amount Due: $1,250.00."
# # 12. Personalized Content Recommendation
# question = "Recommend an article about AI ethics."
# context = "Articles available: 1. AI Ethics in Modern Technology, 2. The Future of Machine Learning..."
# # 13. Content Moderation and Review
# question = "Is this comment respectful?"
# context = "Comment: 'You are so stupid and worthless!'"
# # 14. Healthcare and Medical Queries
# question = "What are the symptoms of diabetes?"
# context = "Diabetes symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision."
# # 15. Customer Feedback Analysis
# question = "What are the main concerns in this feedback?"
# context = "Customers have mentioned that the product often arrives late and the customer service is not responsive."
# # 16. Interactive Storytelling
# question = "What is the hero's main challenge?"
# context = "The hero must overcome his fear of failure to defeat the villain and save the kingdom."
# # 17. Interactive Storytelling
# question = "Does this document comply with GDPR?"
# context = "The document outlines data handling procedures, including consent, data portability, and right to be forgotten."
# # 18. Historical analysis
# context: "In the years leading up to World War II, Europe was a continent marked by political instability, economic turmoil, and the rise of totalitarian regimes. The Treaty of Versailles, signed in 1919, had left Germany in a state of economic depression and national humiliation, which Adolf Hitler and the Nazi Party capitalized on to gain power. Hitler's aggressive expansionist policies, including the annexation of Austria and the invasion of Czechoslovakia, went largely unchecked by the other European powers. The policy of appeasement, championed by British Prime Minister Neville Chamberlain, was intended to prevent another devastating conflict by conceding to some of Hitler's demands. However, this approach ultimately failed to curb Nazi aggression, leading to the outbreak of World War II in 1939. The war brought unprecedented destruction to Europe, with millions of lives lost and cities reduced to rubble. The Holocaust, in which six million Jews were systematically exterminated, remains one of the darkest chapters in human history. The war ended with the unconditional surrender of Germany in May 1945 and the subsequent division of the country into East and West during the Cold War."
# question: "What was the policy of appeasement, and why did it fail?"
# # 19. Scientific Research
# context: "Climate change is a complex global phenomenon with wide-reaching impacts on natural ecosystems, human societies, and the global economy. It is primarily driven by the increase in greenhouse gases in the Earth's atmosphere, particularly carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O), which are largely the result of human activities such as burning fossil fuels, deforestation, and industrial agriculture. These gases trap heat in the atmosphere, leading to a rise in global temperatures, a process known as global warming. The consequences of climate change include rising sea levels due to the melting of polar ice caps, more frequent and severe weather events like hurricanes and droughts, and shifts in ecosystems and biodiversity as species struggle to adapt to changing conditions. Moreover, climate change poses significant risks to human health, agriculture, water resources, and infrastructure. Efforts to mitigate climate change focus on reducing greenhouse gas emissions, transitioning to renewable energy sources, and implementing conservation and sustainable land use practices. International agreements, such as the Paris Agreement, aim to unite countries in the global effort to limit temperature increases and reduce the impact of climate change. However, achieving these goals requires significant political will, technological innovation, and societal change."
# question: "What are the main human activities contributing to climate change, and what are the potential impacts?"
# # 20. Literary Analysis
# context: "Jane Austen's 'Pride and Prejudice' is a novel that explores the themes of love, marriage, social class, and individual growth in early 19th century England. The story centers around Elizabeth Bennet, one of five sisters, as she navigates the expectations of her society and her own personal values. Elizabeth is intelligent, witty, and independent, often challenging the conventional norms of her time. Her initial judgments of people, particularly Mr. Darcy, a wealthy and seemingly arrogant gentleman, are based on first impressions and societal prejudices. As the novel progresses, Elizabeth and Darcy's relationship evolves from mutual disdain to understanding and affection, revealing the importance of personal growth and self-awareness. The novel critiques the rigid social structures and the economic pressures that often dictated marriage choices during this period. Through the development of its characters and their interactions, 'Pride and Prejudice' offers a nuanced portrayal of the complexities of human relationships and the struggles of individuals to reconcile personal happiness with societal expectations. Austen's use of irony, free indirect speech, and sharp social commentary make 'Pride and Prejudice' a timeless work of literature that continues to resonate with readers today."
# question: "How does Jane Austen critique the social class and marriage conventions of her time through the characters in 'Pride and Prejudice'?"

