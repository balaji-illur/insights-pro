import time

import chromadb
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


class ChinookMetadataSystem:
    def __init__(self, embedding_model="models/gemini-embedding-001"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.chroma_client = chromadb.Client()
        self.collection_name = "chinook_metadata"
        self._initialize_metadata()

    def _initialize_metadata(self):
        """Initialize and populate the vector database with Chinook metadata"""
        documents = []

        # 1. TABLE-LEVEL CHUNKS
        table_chunks = self._create_table_chunks()
        documents.extend(table_chunks)

        # 2. COLUMN-LEVEL CHUNKS
        column_chunks = self._create_column_chunks()
        documents.extend(column_chunks)

        # 3. RELATIONSHIP CHUNKS
        relationship_chunks = self._create_relationship_chunks()
        documents.extend(relationship_chunks)

        # 4. QUERY PATTERN CHUNKS
        pattern_chunks = self._create_pattern_chunks()
        documents.extend(pattern_chunks)

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents, embedding=self.embeddings, collection_name=self.collection_name
        )

    def _create_table_chunks(self) -> list[Document]:
        """Create semantic chunks for each table"""
        tables = {
            "employees": {
                "description": "Employee data with hierarchical reporting structure",
                "business_purpose": "HR management, organizational hierarchy, employee performance tracking",
                "common_queries": "manager reports, employee lookup, organizational structure",
            },
            "customers": {
                "description": "Customer demographic and contact information",
                "business_purpose": "Customer relationship management, billing, demographics analysis",
                "common_queries": "customer lookup, regional analysis, support contact",
            },
            "invoices": {
                "description": "Invoice header data with customer and billing information",
                "business_purpose": "Sales tracking, revenue analysis, customer billing",
                "common_queries": "sales reports, customer purchase history, revenue analysis",
            },
            "invoice_items": {
                "description": "Individual line items for each invoice with track details",
                "business_purpose": "Detailed sales analysis, track popularity, pricing analysis",
                "common_queries": "track sales, detailed purchase analysis, quantity tracking",
            },
            "artists": {
                "description": "Music artist information and catalog management",
                "business_purpose": "Artist catalog, music library organization, royalty management",
                "common_queries": "artist lookup, catalog browsing, artist performance",
            },
            "albums": {
                "description": "Album catalog with artist relationships",
                "business_purpose": "Music catalog management, album sales tracking, artist discography",
                "common_queries": "discography, album search, artist albums",
            },
            "tracks": {
                "description": "Individual song/track data with album, genre, and media type",
                "business_purpose": "Music library, track analysis, playlist management, sales tracking",
                "common_queries": "song search, music analysis, track popularity, playlist creation",
            },
            "genres": {
                "description": "Music genre classification system",
                "business_purpose": "Music categorization, genre-based analysis, recommendation systems",
                "common_queries": "genre analysis, music categorization, preference tracking",
            },
            "media_types": {
                "description": "Audio file format specifications",
                "business_purpose": "Technical format management, quality analysis, compatibility",
                "common_queries": "format analysis, technical specifications, file type filtering",
            },
            "playlists": {
                "description": "User-created music playlists",
                "business_purpose": "Music curation, user preferences, listening patterns",
                "common_queries": "playlist management, user preferences, music curation",
            },
            "playlist_track": {
                "description": "Many-to-many relationship between playlists and tracks",
                "business_purpose": "Playlist composition, track associations, music recommendation",
                "common_queries": "playlist contents, track associations, music relationships",
            },
        }

        chunks = []
        for table_name, info in tables.items():
            content = f"Table: {table_name}\nDescription: {info['description']}\nBusiness Purpose: {info['business_purpose']}\nCommon Use Cases: {info['common_queries']}"

            chunks.append(
                Document(
                    page_content=content, metadata={"type": "table", "table_name": table_name, "category": "schema"}
                )
            )

        return chunks

    def _create_column_chunks(self) -> list[Document]:
        """Create chunks for individual columns with business context"""
        columns = {
            # Employees
            "employees.EmployeeId": "INTEGER PRIMARY KEY - Unique employee identifier for HR systems",
            "employees.LastName": "NVARCHAR(20) - Employee surname for formal identification",
            "employees.FirstName": "NVARCHAR(20) - Employee given name for personal identification",
            "employees.Title": "NVARCHAR(30) - Job title and position within organization",
            "employees.ReportsTo": "INTEGER FOREIGN KEY - Manager relationship, references employees.EmployeeId for org hierarchy",
            "employees.BirthDate": "DATETIME - Employee birth date for HR compliance and demographics",
            "employees.HireDate": "DATETIME - Employment start date for tenure analysis",
            "employees.Address": "NVARCHAR(70) - Employee street address",
            "employees.City": "NVARCHAR(40) - Employee city",
            "employees.State": "NVARCHAR(40) - Employee state",
            "employees.Country": "NVARCHAR(40) - Employee country",
            "employees.PostalCode": "NVARCHAR(10) - Employee postal code",
            "employees.Phone": "NVARCHAR(24) - Employee phone number",
            "employees.Fax": "NVARCHAR(24) - Employee fax number",
            "employees.Email": "NVARCHAR(60) - Employee email address",
            # Customers
            "customers.CustomerId": "INTEGER PRIMARY KEY - Unique customer identifier for CRM",
            "customers.FirstName": "NVARCHAR(40) - Customer personal name for relationship management",
            "customers.LastName": "NVARCHAR(20) - Customer family name for formal correspondence",
            "customers.Company": "NVARCHAR(80) - Business affiliation for B2B relationships",
            "customers.Address": "NVARCHAR(70) - Street address for shipping and billing",
            "customers.City": "NVARCHAR(40) - City location for regional analysis",
            "customers.State": "NVARCHAR(40) - State/province for geographic segmentation",
            "customers.Country": "NVARCHAR(40) - Country for international market analysis",
            "customers.PostalCode": "NVARCHAR(10) - Postal code for precise location targeting",
            "customers.Phone": "NVARCHAR(24) - Contact phone for customer service",
            "customers.Fax": "NVARCHAR(24) - Fax number for business communications",
            "customers.Email": "NVARCHAR(60) - Email address for digital marketing and support",
            "customers.SupportRepId": "INTEGER FOREIGN KEY - Assigned support representative",
            # Invoices
            "invoices.InvoiceId": "INTEGER PRIMARY KEY - Unique invoice identifier for billing",
            "invoices.CustomerId": "INTEGER FOREIGN KEY - Customer who made purchase, references customers",
            "invoices.InvoiceDate": "DATETIME - Purchase date for sales analysis and trends",
            "invoices.BillingAddress": "NVARCHAR(70) - Invoice billing address",
            "invoices.BillingCity": "NVARCHAR(40) - Billing city for tax and regional analysis",
            "invoices.BillingState": "NVARCHAR(40) - Billing state for tax and regional analysis",
            "invoices.BillingCountry": "NVARCHAR(40) - Billing country for tax and regional analysis",
            "invoices.BillingPostalCode": "NVARCHAR(10) - Billing postal code for tax and regional analysis",
            "invoices.Total": "NUMERIC - Total invoice amount for revenue analysis",
            # Invoice Items
            "invoice_items.InvoiceLineId": "INTEGER PRIMARY KEY - Unique line item identifier",
            "invoice_items.InvoiceId": "INTEGER FOREIGN KEY - Parent invoice reference",
            "invoice_items.TrackId": "INTEGER FOREIGN KEY - Purchased track reference",
            "invoice_items.UnitPrice": "NUMERIC - Individual track price for pricing analysis",
            "invoice_items.Quantity": "INTEGER - Number of items purchased",
            # Artists
            "artists.ArtistId": "INTEGER PRIMARY KEY - Unique artist identifier for music catalog",
            "artists.Name": "NVARCHAR(120) - Artist or band name for music browsing",
            # Albums
            "albums.AlbumId": "INTEGER PRIMARY KEY - Unique album identifier for catalog management",
            "albums.Title": "NVARCHAR(160) - Album title for music library organization",
            "albums.ArtistId": "INTEGER FOREIGN KEY - Album creator, references artists table",
            # Tracks
            "tracks.TrackId": "INTEGER PRIMARY KEY - Unique track identifier for music library",
            "tracks.Name": "NVARCHAR(200) - Song title for music search and browsing",
            "tracks.AlbumId": "INTEGER FOREIGN KEY - Parent album reference for discography",
            "tracks.MediaTypeId": "INTEGER FOREIGN KEY - Audio format specification",
            "tracks.GenreId": "INTEGER FOREIGN KEY - Music genre classification",
            "tracks.Composer": "NVARCHAR(220) - Song composer for music credits and royalties",
            "tracks.Milliseconds": "INTEGER - Track duration for playlist and analysis",
            "tracks.Bytes": "INTEGER - File size for storage and streaming analysis",
            "tracks.UnitPrice": "NUMERIC - Track selling price for revenue analysis",
            # Genres
            "genres.GenreId": "INTEGER PRIMARY KEY - Unique genre identifier for classification",
            "genres.Name": "NVARCHAR(120) - Genre name for music categorization and filtering",
            # Media Types
            "media_types.MediaTypeId": "INTEGER PRIMARY KEY - Unique media format identifier",
            "media_types.Name": "NVARCHAR(120) - Format specification like MP3, AAC for technical analysis",
            # Playlists
            "playlists.PlaylistId": "INTEGER PRIMARY KEY - Unique playlist identifier for curation",
            "playlists.Name": "NVARCHAR(120) - Playlist title for organization and browsing",
            # Playlist Track Junction
            "playlist_track.PlaylistId": "INTEGER FOREIGN KEY - Playlist reference for many-to-many relationship",
            "playlist_track.TrackId": "INTEGER FOREIGN KEY - Track reference for playlist composition",
        }

        chunks = []
        for col_name, description in columns.items():
            table_name = col_name.split(".")[0]
            column_name = col_name.split(".")[1]

            chunks.append(
                Document(
                    page_content=f"Column: {col_name}\nDetails: {description}",
                    metadata={
                        "type": "column",
                        "table_name": table_name,
                        "column_name": column_name,
                        "full_name": col_name,
                    },
                )
            )

        return chunks

    def _create_relationship_chunks(self) -> list[Document]:
        """Create chunks for foreign key relationships and business logic"""
        relationships = [
            {
                "description": "employees.ReportsTo → employees.EmployeeId creates organizational hierarchy for manager-subordinate relationships",
                "tables": ["employees"],
                "relationship_type": "self_referencing",
                "business_logic": "hierarchical reporting structure",
            },
            {
                "description": "customers.SupportRepId → employees.EmployeeId assigns customer support representatives",
                "tables": ["customers", "employees"],
                "relationship_type": "one_to_many",
                "business_logic": "customer service assignment",
            },
            {
                "description": "invoices.CustomerId → customers.CustomerId links purchases to customers",
                "tables": ["invoices", "customers"],
                "relationship_type": "one_to_many",
                "business_logic": "customer purchase tracking",
            },
            {
                "description": "invoice_items.InvoiceId → invoices.InvoiceId connects line items to invoice headers",
                "tables": ["invoice_items", "invoices"],
                "relationship_type": "one_to_many",
                "business_logic": "detailed invoice breakdown",
            },
            {
                "description": "invoice_items.TrackId → tracks.TrackId identifies purchased music tracks",
                "tables": ["invoice_items", "tracks"],
                "relationship_type": "many_to_one",
                "business_logic": "track sales analysis",
            },
            {
                "description": "albums.ArtistId → artists.ArtistId connects albums to their creators",
                "tables": ["albums", "artists"],
                "relationship_type": "many_to_one",
                "business_logic": "artist discography management",
            },
            {
                "description": "tracks.AlbumId → albums.AlbumId organizes songs within albums",
                "tables": ["tracks", "albums"],
                "relationship_type": "many_to_one",
                "business_logic": "album track listing",
            },
            {
                "description": "tracks.GenreId → genres.GenreId categorizes music by genre",
                "tables": ["tracks", "genres"],
                "relationship_type": "many_to_one",
                "business_logic": "music genre classification",
            },
            {
                "description": "tracks.MediaTypeId → media_types.MediaTypeId specifies audio format",
                "tables": ["tracks", "media_types"],
                "relationship_type": "many_to_one",
                "business_logic": "technical format specification",
            },
            {
                "description": "playlist_track.PlaylistId → playlists.PlaylistId + playlist_track.TrackId → tracks.TrackId creates many-to-many playlist-track relationship",
                "tables": ["playlist_track", "playlists", "tracks"],
                "relationship_type": "many_to_many",
                "business_logic": "playlist composition and track associations",
            },
        ]

        chunks = []
        for rel in relationships:
            content = f"Relationship: {rel['description']}\nType: {rel['relationship_type']}\nBusiness Logic: {rel['business_logic']}"

            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "type": "relationship",
                        "tables": ", ".join(rel["tables"]),
                        "relationship_type": rel["relationship_type"],
                    },
                )
            )

        return chunks

    def _create_pattern_chunks(self) -> list[Document]:
        """Create chunks for common query patterns and business scenarios"""
        patterns = [
            {
                "pattern": "Customer purchase analysis requires invoices → customers join for customer demographics",
                "tables": ["invoices", "customers"],
                "query_type": "customer_analysis",
            },
            {
                "pattern": "Track sales analysis needs invoice_items → tracks → albums → artists chain for complete music hierarchy",
                "tables": ["invoice_items", "tracks", "albums", "artists"],
                "query_type": "sales_analysis",
            },
            {
                "pattern": "Employee performance uses invoices → customers → employees through SupportRepId for sales attribution",
                "tables": ["invoices", "customers", "employees"],
                "query_type": "employee_performance",
            },
            {
                "pattern": "Music catalog browsing requires tracks → albums → artists → genres → media_types for complete metadata",
                "tables": ["tracks", "albums", "artists", "genres", "media_types"],
                "query_type": "catalog_browsing",
            },
            {
                "pattern": "Playlist analysis uses playlist_track junction to connect playlists ↔ tracks many-to-many",
                "tables": ["playlists", "playlist_track", "tracks"],
                "query_type": "playlist_analysis",
            },
            {
                "pattern": "Revenue analysis combines invoices.Total with invoice_items for detailed and summary views",
                "tables": ["invoices", "invoice_items"],
                "query_type": "revenue_analysis",
            },
            {
                "pattern": "Geographic analysis uses customers.Country, State, City with invoices for regional sales",
                "tables": ["customers", "invoices"],
                "query_type": "geographic_analysis",
            },
            {
                "pattern": "Time-based analysis uses invoices.InvoiceDate for trends, seasonal patterns, growth analysis",
                "tables": ["invoices"],
                "query_type": "temporal_analysis",
            },
        ]

        chunks = []
        for pattern in patterns:
            chunks.append(
                Document(
                    page_content=pattern["pattern"],
                    metadata={
                        "type": "pattern",
                        "tables": ", ".join(pattern["tables"]),
                        "query_type": pattern["query_type"],
                    },
                )
            )

        return chunks

    def analyze_question(self, question: str) -> dict:
        """Extract key entities and query intent from user question"""
        question_lower = question.lower()

        # Entity extraction
        table_keywords = {
            "employees": ["employee", "staff", "worker", "manager", "supervisor", "team", "hire", "report"],
            "customers": ["customer", "client", "buyer", "user", "purchaser", "demographic"],
            "invoices": ["invoice", "bill", "purchase", "order", "transaction", "sale", "payment"],
            "invoice_items": ["item", "line item", "product", "detail", "quantity"],
            "artists": ["artist", "band", "musician", "performer", "singer", "composer"],
            "albums": ["album", "record", "release", "discography", "collection"],
            "tracks": ["track", "song", "music", "audio", "recording", "tune"],
            "genres": ["genre", "style", "category", "type", "classification"],
            "media_types": ["format", "media", "file type", "audio format", "mp3", "aac"],
            "playlists": ["playlist", "collection", "mix", "compilation"],
        }

        detected_tables = []
        for table, keywords in table_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_tables.append(table)

        # Intent detection
        intent_patterns = {
            "aggregation": ["total", "sum", "count", "average", "max", "min", "most", "least", "top"],
            "filtering": ["where", "filter", "specific", "only", "exclude", "include"],
            "temporal": ["date", "time", "year", "month", "recent", "latest", "since", "between"],
            "ranking": ["top", "best", "worst", "rank", "order", "sort", "popular"],
            "comparison": ["compare", "versus", "vs", "difference", "more than", "less than"],
            "geographical": ["country", "city", "state", "region", "location", "geographic"],
        }

        detected_intents = []
        for intent, patterns in intent_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                detected_intents.append(intent)

        return {"detected_tables": detected_tables, "detected_intents": detected_intents, "question": question}

    def retrieve_relevant_metadata(self, question: str, max_tokens: int = 2000) -> dict:
        """Multi-stage retrieval of relevant metadata"""

        # Step 1: Analyze question
        analysis = self.analyze_question(question)

        # Step 2: Primary vector retrieval
        primary_results = self.vectorstore.similarity_search(question, k=15, filter=None)

        # Step 3: Entity expansion - get all columns for detected tables
        expansion_results = []
        if analysis["detected_tables"]:
            for table in analysis["detected_tables"]:
                table_results = self.vectorstore.similarity_search(
                    f"table {table} columns", k=10, filter={"table_name": table}
                )
                expansion_results.extend(table_results)

        # Step 4: Relationship traversal
        relationship_results = self.vectorstore.similarity_search(question, k=8, filter={"type": "relationship"})

        # Step 5: Combine and deduplicate
        all_results = primary_results + expansion_results + relationship_results
        seen_content = set()
        unique_results = []

        for doc in all_results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_results.append(doc)

        # Step 6: Intelligent filtering by token budget
        # filtered_results = self._apply_token_budget(unique_results, max_tokens)

        return {"relevant_documents": unique_results, "analysis": analysis, "total_chunks": len(unique_results)}

    def _apply_token_budget(self, documents: list[Document], max_tokens: int) -> list[Document]:
        """Filter documents to stay within token budget while prioritizing essential metadata"""

        # Prioritize by type: tables > relationships > columns > patterns
        priority_order = {"table": 4, "relationship": 3, "column": 2, "pattern": 1}

        # Sort by priority and relevance
        sorted_docs = sorted(documents, key=lambda x: priority_order.get(x.metadata.get("type", ""), 0), reverse=True)

        # Estimate tokens (rough: 4 chars per token)
        current_tokens = 0
        filtered_docs = []

        for doc in sorted_docs:
            doc_tokens = len(doc.page_content) // 4
            if current_tokens + doc_tokens <= max_tokens:
                filtered_docs.append(doc)
                current_tokens += doc_tokens
            else:
                break

        return filtered_docs

    def construct_dynamic_prompt(self, question: str) -> str:
        """Generate dynamic prompt with relevant metadata context"""

        # Retrieve relevant metadata
        metadata_result = self.retrieve_relevant_metadata(question)
        docs = metadata_result["relevant_documents"]

        # Organize retrieved content by type
        tables = [d for d in docs if d.metadata.get("type") == "table"]
        columns = [d for d in docs if d.metadata.get("type") == "column"]
        relationships = [d for d in docs if d.metadata.get("type") == "relationship"]
        patterns = [d for d in docs if d.metadata.get("type") == "pattern"]

        # Construct prompt sections
        prompt_sections = []

        prompt_sections.append("You are an expert SQL analyst for the Chinook music database.")

        if tables:
            prompt_sections.append("\n## RELEVANT TABLES:")
            for table_doc in tables:
                prompt_sections.append(table_doc.page_content)

        if columns:
            prompt_sections.append("\n## RELEVANT COLUMNS:")
            for col_doc in columns:
                prompt_sections.append(col_doc.page_content)

        if relationships:
            prompt_sections.append("\n## KEY RELATIONSHIPS:")
            for rel_doc in relationships:
                prompt_sections.append(rel_doc.page_content)

        if patterns:
            prompt_sections.append("\n## RELEVANT QUERY PATTERNS:")
            for pattern_doc in patterns:
                prompt_sections.append(pattern_doc.page_content)

        prompt_sections.append(f"\n## USER QUESTION:\n{question}")

        prompt_sections.append("\n## INSTRUCTIONS:")
        prompt_sections.append("Generate a valid SQL query that accurately answers the user's question.")
        prompt_sections.append("Use only the tables and columns provided in the context above.")
        prompt_sections.append("Include proper JOIN conditions based on the relationships specified.")
        prompt_sections.append("Return only the SQL query without explanation.")

        return "\n".join(prompt_sections)


# Usage Example
def main():
    # Initialize the system
    metadata_system = ChinookMetadataSystem()

    # Example questions
    test_questions = ["which country bought the highest numbers of sales? also list down top 5 countries"]

    for question in test_questions:
        print(f"\n{'=' * 50}")
        print(f"Question: {question}")
        print(f"{'=' * 50}")

        try:
            time.sleep(3)
            prompt = metadata_system.construct_dynamic_prompt(question)
            print("Generated Prompt:")
            print(prompt)

        except Exception as e:
            print(f"Error processing question: {e!s}")


if __name__ == "__main__":
    main()
