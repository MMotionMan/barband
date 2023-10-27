import grpc
import grpc.experimental
from concurrent import futures

from protobufs.grpc_change_index_pb2_grpc import ChangeIndexServicer, add_ChangeIndexServicer_to_server


class ChangeServicer(ChangeIndexServicer):
    def Change(self, request, context):
        return super().Change(request, context)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ChangeIndexServicer_to_server(ChangeServicer(), server)
    server.add_insecure_port('localhost:50051')
    server.start()
    server.wait_for_termination()

serve()