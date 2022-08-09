from math import ceil
import os

class File():
    def __init__(self, path : str, chunk_size: int):
        self.path = path
        self.name = path.split('/')[-1]
        self.file_size = os.path.getsize(path)

        self.chunk_size = chunk_size
        self.chunks_number = ceil(float(self.file_size) / float(self.chunk_size))

    def get_chunks_number(self) -> int :
        return self.chunks_number

    def get_name (self) -> str:
        return self.name
 
    def get_path(self) -> str :
        return self.path
    
    def get_chunks_number(self) -> int:
        return self.chunks_number
    
    def get_chunk_size(self) -> int:
        return self.chunk_size

    def get_file_size(self) -> int :
        return self.file_size