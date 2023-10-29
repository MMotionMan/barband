import grpc
import grpc.experimental
from concurrent import futures

from barband_ml.index_operations import VectorChanger
from barband_ml.word2vec_model import Word2Vec
import protobufs.grpc_change_index_pb2_grpc as pb2_grpc
import protobufs.grpc_change_index_pb2 as pb2


class ChangeServicer(pb2_grpc.ChangeIndexServicer):
    def __init__(self):
        pass

    def Change(self, request, context):
        try:
            action = request.action
            result = {'ErrorMessage': "I don't have action type"}
            word2vec_model = Word2Vec()
            vector_changer = VectorChanger("/Users/anatoliy/PycharmProjects/barband/barband_ml/online_flat.index",
                                           "/Users/anatoliy/PycharmProjects/barband/barband_ml/query.index")
            if action == 0:
                for event in request.events:
                    vector = word2vec_model.text_to_tensor(event, event.language)
                    vector_changer.add_to_vector(vector, event.id)

                result = {'ErrorMessage': f'{len(request.events)} records added'}
            elif action == 1:
                for event in request.events:
                    vector_changer.delete_from_vector(event.id)

                result = {'ErrorMessage': f'{len(request.events)} records deleted'}
            elif action == 2:
                vector_changer.update_index()

                result = {'ErrorMessage': 'Vector updated'}
            # print("action =", action)
            # print(message, f'\\n {request.events[0].description}')

            return pb2.Response(**result)

        except:
            result = {'ErrorMessage': "Error"}
            return pb2.Response(**result)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ChangeIndexServicer_to_server(ChangeServicer(), server)
    server.add_insecure_port('localhost:50051')
    server.start()
    server.wait_for_termination()


serve()
