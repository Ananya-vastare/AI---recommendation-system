from django.db import models

class UserSearch(models.Model):
    query_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.query_text