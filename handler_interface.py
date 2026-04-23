class Handler:
    def __init__(self):
        self.next_handler = None

    #Combine this handler to the next handler    
    def set_next(self, handler):
        self.next_handler = handler
        return handler
    
    #Combine/pass on data to th next handler, if there is one, otherwise return the data
    def set_next(self, handler):
        self.next_handler = handler
        return handler