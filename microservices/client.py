from protobufs import grpc_change_index_pb2, grpc_change_index_pb2_grpc

import grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = grpc_change_index_pb2_grpc.ChangeIndexStub(channel)
        print('1. sayHello')
        request = grpc_change_index_pb2.DESCRIPTOR

run()