from utils.processor import Processor

rpc_nodes = [
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al5vgPg9eKgR74_chlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al4IyDM-d_sR74_ChlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=Asv3nQwPQE6AqrmVKYZBEgAFdL22d-8R74-_hlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6WvxNheKgR74_dhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6oHm4xeKgR74_ehlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6vcdfheKgR74_fhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al63O0vVeKgR74_ghlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al6_PDK6eKgR74_hhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al7S1iypeKgR74_jhlDYfw4q',
    'https://lb.drpc.org/ogrpc?network=ethereum&dkey=AslpqdMF10aJj7M_sfS8al7c4vJFeKgR74_khlDYfw4q',
]

rpc_res_node = []


def create_processor(retry=5):
    for i in range(retry):
        try:
            for node in rpc_nodes:
                processor = Processor(chain=None, transaction=None, rpc_node=node)
                if processor.w3.is_connected():
                    rpc_res_node.append(node)
        except Exception as e:
            print(node)
    return rpc_res_node


if __name__ == '__main__':
    res = create_processor()
    print(list(set(res)))
