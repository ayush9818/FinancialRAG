import warnings
warnings.filterwarnings("ignore")

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
import faiss
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import QueryBundle


# TODO: Create Custom Retriever Class after finalizing the experiment
# https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers/
class RetrievalAgent:
    def __init__(self, cfg, documents):
        self.cfg = cfg 
        self.documents = documents 

        self.index , self.reranker = self.initialise_retrieval_components()

    def initialise_retrieval_components(self):
        # Create FaisVectorStore to store embeddings
        fais_index = faiss.IndexFlatL2(self.cfg.EMBED_DIMENSION)
        vector_store = FaissVectorStore(faiss_index=fais_index)
        print("Vector Store Created")

        ## Can experiment with different transformations
        base_pipeline = IngestionPipeline(
            # chunk_size=256, chunk_overlap=20
            transformations=[SentenceSplitter()],
            vector_store=vector_store,
            documents=self.documents
        )
        nodes = base_pipeline.run()

        # Create vector index from base nodes
        index = VectorStoreIndex(nodes)
        print("Vector Index Initialised")
        
        # Create Reranker
        reranker = SentenceTransformerRerank(
                    model=self.cfg.RERANKER_MODEL,
                    top_n=self.cfg.RERANKER_TOP_N
                )
        print("Reranker Initialised")
        return index, reranker 

    def retrieve_nodes(self, query_str, with_reranker=True):
        query_bundle = QueryBundle(query_str)
        # configure retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.cfg.SIM_TOP_K
        )
        retrieved_nodes = retriever.retrieve(query_bundle)

        if with_reranker:    
            retrieved_nodes = self.reranker.postprocess_nodes(
                retrieved_nodes, query_bundle
            )

        return retrieved_nodes