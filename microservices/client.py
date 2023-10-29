from protobufs import grpc_change_index_pb2, grpc_change_index_pb2_grpc

import grpc


class UnaryClient(object):
    """
    Client for gRPC functionality
    """

    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051

        # instantiate a channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))

        # bind the client and the server
        self.stub = grpc_change_index_pb2_grpc.ChangeIndexStub(self.channel)

    def get_url(self, events, action):
        """
        Client function to call the rpc for GetServerResponse
        """
        message = grpc_change_index_pb2.RequestToIndex(events=events, action=action)
        print(f'{message}')
        return self.stub.Change(message)


if __name__ == '__main__':
    client = UnaryClient()
    message = grpc_change_index_pb2.EventInfo(description='Первое мероприятие',
                                              title='Название мероприятия',
                                              language='Русский',
                                              tags=['Первый тег', 'Второй тег'],
                                              categories=['1-я Категория', '2-я Категория'],
                                              category_level=[1, 1])
    messages = [message, message]
    action = 0
    result = client.get_url(events=messages, action=action)
    print(f'{result}')
