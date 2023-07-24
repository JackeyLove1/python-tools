class Document:
   def __init__(self, query, answer):
       self.query = query
       self.answer = answer

doc = Document("q", "a")
print(doc.query)
print(doc.answer)