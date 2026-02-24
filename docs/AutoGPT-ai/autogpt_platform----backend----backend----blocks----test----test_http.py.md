
# `.\AutoGPT\autogpt_platform\backend\backend\blocks\test\test_http.py` 详细设计文档

This file contains a test suite for the HTTP block integration with HostScopedCredentials, testing various scenarios such as exact host matching, wildcard matching, non-matching credentials, user headers overriding credential headers, auto-discovery of credentials, and multiple header credentials.

## 整体流程

```mermaid
graph TD
    A[Start] --> B[Create SendAuthenticatedWebRequestBlock instance]
    B --> C[Create mock_response]
    C --> D[Create exact_match_credentials]
    D --> E[Create wildcard_credentials]
    E --> F[Create non_matching_credentials]
    F --> G[Run test_http_block_with_exact_host_match]
    G --> H[Run test_http_block_with_wildcard_host_match]
    H --> I[Run test_http_block_with_non_matching_credentials]
    I --> J[Run test_user_headers_override_credential_headers]
    J --> K[Run test_auto_discovered_credentials_flow]
    K --> L[Run test_multiple_header_credentials]
    L --> M[Run test_credentials_with_complex_url_patterns]
    M --> N[End]
```

## 类结构

```
TestHttpBlockWithHostScopedCredentials (测试类)
├── make_test_context (函数)
│   ├── graph TD
│   ├── A[开始]
│   ├── B{金额 > 0?}
│   ├── B -- 否 --> C[抛出 ValueError]
│   ├── B -- 是 --> D[调用第三方支付网关]
│   ├── D --> E{网关返回成功?}
│   ├── E -- 是 --> F[更新订单状态]
│   └── E -- 否 --> G[记录日志并重试]
└── test_http_block_with_exact_host_match (测试函数)
    ├── mock_requests_class (全局变量)
    ├── http_block (全局变量)
    ├── exact_match_credentials (全局变量)
    ├── mock_response (全局变量)
    ├── result (局部变量)
    ├── input_data (局部变量)
    ├── output_name (局部变量)
    └── output_data (局部变量)
```

## 全局变量及字段


### `mock_requests_class`
    
Mocked Requests class used for testing HTTP requests.

类型：`MagicMock`
    


### `http_block`
    
Instance of the HTTP block to be tested.

类型：`SendAuthenticatedWebRequestBlock`
    


### `exact_match_credentials`
    
Host-scoped credentials for exact domain matching.

类型：`HostScopedCredentials`
    


### `mock_response`
    
Mocked HTTP response object used for testing.

类型：`MagicMock`
    


### `result`
    
List of tuples containing output names and data from the HTTP block run.

类型：`list`
    


### `input_data`
    
Input data for the HTTP block run.

类型：`SendAuthenticatedWebRequestBlock.Input`
    


### `output_name`
    
Name of the output from the HTTP block run.

类型：`str`
    


### `output_data`
    
Data of the output from the HTTP block run.

类型：`dict`
    


### `TestHttpBlockWithHostScopedCredentials.http_block`
    
HTTP block instance to be tested.

类型：`SendAuthenticatedWebRequestBlock`
    


### `TestHttpBlockWithHostScopedCredentials.mock_requests_class`
    
Mocked Requests class used for testing HTTP requests.

类型：`MagicMock`
    


### `TestHttpBlockWithHostScopedCredentials.exact_match_credentials`
    
Host-scoped credentials for exact domain matching.

类型：`HostScopedCredentials`
    


### `TestHttpBlockWithHostScopedCredentials.mock_response`
    
Mocked HTTP response object used for testing.

类型：`MagicMock`
    


### `TestHttpBlockWithHostScopedCredentials.result`
    
List of tuples containing output names and data from the HTTP block run.

类型：`list`
    


### `TestHttpBlockWithHostScopedCredentials.input_data`
    
Input data for the HTTP block run.

类型：`SendAuthenticatedWebRequestBlock.Input`
    


### `TestHttpBlockWithHostScopedCredentials.output_name`
    
Name of the output from the HTTP block run.

类型：`str`
    


### `TestHttpBlockWithHostScopedCredentials.output_data`
    
Data of the output from the HTTP block run.

类型：`dict`
    
    

## 全局函数及方法


### make_test_context

Helper to create test ExecutionContext.

参数：

- `graph_exec_id`：`str`，The execution ID of the graph.
- `user_id`：`str`，The user ID for the execution context.

返回值：`ExecutionContext`，A test ExecutionContext object.

#### 流程图

```mermaid
graph TD
    A[make_test_context] --> B[Create ExecutionContext]
    B --> C[Return ExecutionContext]
```

#### 带注释源码

```python
def make_test_context(
    graph_exec_id: str = "test-exec-id",
    user_id: str = "test-user-id",
) -> ExecutionContext:
    """Helper to create test ExecutionContext."""
    return ExecutionContext(
        user_id=user_id,
        graph_exec_id=graph_exec_id,
    )
```



### test_http_block_with_exact_host_match

This function tests the HTTP block with exact host matching credentials.

参数：

- `mock_requests_class`: `MagicMock`，Mocked Requests class for testing
- `http_block`: `SendAuthenticatedWebRequestBlock`，HTTP block instance to be tested
- `exact_match_credentials`: `HostScopedCredentials`，Host-scoped credentials for exact domain matching
- `mock_response`: `MagicMock(spec=Response)`，Mocked successful HTTP response

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with credentials]
    D --> E[Verify request headers]
    E --> F[Verify response handling]
    F --> G[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_http_block_with_exact_host_match(
    self,
    mock_requests_class,
    http_block,
    exact_match_credentials,
    mock_response,
):
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Prepare input data
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.GET,
        headers={"User-Agent": "test-agent"},
        credentials=cast(
            HttpCredentials,
            {
                "id": exact_match_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": exact_match_credentials.title,
            },
        ),
    )

    # Execute with credentials provided by execution manager
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=exact_match_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify request headers include both credential and user headers
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {
        "Authorization": "Bearer exact-match-token",
        "X-API-Key": "api-key-123",
        "User-Agent": "test-agent",
    }
    assert call_args.kwargs["headers"] == expected_headers

    # Verify response handling
    assert len(result) == 1
    assert result[0][0] == "response"
    assert result[0][1] == {"success": True, "data": "test"}
```



### `test_http_block_with_wildcard_host_match`

This function tests the HTTP block with wildcard host pattern matching credentials.

参数：

- `mock_requests_class`：`AsyncMock`，Mocked Requests class for testing
- `http_block`：`SendAuthenticatedWebRequestBlock`，HTTP block instance to be tested
- `wildcard_credentials`：`HostScopedCredentials`，Wildcard host pattern credentials
- `mock_response`：`MagicMock(spec=Response)`，Mocked successful HTTP response

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with wildcard credentials]
    D --> E[Verify wildcard matching works]
    E --> F[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_http_block_with_wildcard_host_match(
    self,
    mock_requests_class,
    http_block,
    wildcard_credentials,
    mock_response,
):
    """Test HTTP block with wildcard host pattern matching."""
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with subdomain that should match *.github.com
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.github.com/user",
        method=HttpMethod.GET,
        headers={},
        credentials=cast(
            HttpCredentials,
            {
                "id": wildcard_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": wildcard_credentials.title,
            },
        ),
    )

    # Execute with wildcard credentials
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=wildcard_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify wildcard matching works
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {"Authorization": "token ghp_wildcard123"}
    assert call_args.kwargs["headers"] == expected_headers
``` 



### test_http_block_with_non_matching_credentials

This function tests the HTTP block when the credentials do not match the target URL.

参数：

- `mock_requests_class`: `MagicMock`，A mock of the Requests class used to simulate HTTP requests.
- `http_block`: `SendAuthenticatedWebRequestBlock`，An instance of the HTTP block to be tested.
- `non_matching_credentials`: `HostScopedCredentials`，Credentials that do not match the test URLs.
- `mock_response`: `MagicMock(spec=Response)`，A mock of the HTTP response object.

返回值：`None`，This function does not return a value.

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with non-matching credentials]
    D --> E[Verify only user headers are included]
    E --> F[End]
```

#### 带注释源码

```python
@patch("backend.blocks.http.Requests")
async def test_http_block_with_non_matching_credentials(
    self,
    mock_requests_class,
    http_block,
    non_matching_credentials,
    mock_response,
):
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with URL that doesn't match the credentials
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.GET,
        headers={"User-Agent": "test-agent"},
        credentials=cast(
            HttpCredentials,
            {
                "id": non_matching_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": non_matching_credentials.title,
            },
        ),
    )

    # Execute with non-matching credentials
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=non_matching_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify only user headers are included (no credential headers)
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {"User-Agent": "test-agent"}
    assert call_args.kwargs["headers"] == expected_headers
``` 



### test_user_headers_override_credential_headers

This function tests that user-provided headers take precedence over credential headers when making an HTTP request.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class used to simulate HTTP requests.
- `http_block`：`SendAuthenticatedWebRequestBlock`，Instance of the HTTP block to be tested.
- `exact_match_credentials`：`HostScopedCredentials`，Credentials for the exact host matching.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.

返回值：`None`，This function does not return any value.

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with conflicting headers]
    D --> E[Verify user headers take precedence]
    E --> F[End]
```

#### 带注释源码

```python
@patch("backend.blocks.http.Requests")
async def test_user_headers_override_credential_headers(
    self,
    mock_requests_class,
    http_block,
    exact_match_credentials,
    mock_response,
):
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with user header that conflicts with credential header
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.POST,
        headers={
            "Authorization": "Bearer user-override-token",  # Should override
            "Content-Type": "application/json",  # Additional user header
        },
        credentials=cast(
            HttpCredentials,
            {
                "id": exact_match_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": exact_match_credentials.title,
            },
        ),
    )

    # Execute with conflicting headers
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=exact_match_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify user headers take precedence
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {
        "X-API-Key": "api-key-123",  # From credentials
        "Authorization": "Bearer user-override-token",  # User override
        "Content-Type": "application/json",  # User header
    }
    assert call_args.kwargs["headers"] == expected_headers
```



### test_auto_discovered_credentials_flow

This function tests the auto-discovery flow where the execution manager provides matching credentials.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing HTTP requests.
- `http_block`：`SendAuthenticatedWebRequestBlock`，The HTTP block instance to be tested.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.
- `auto_discovered_creds`：`HostScopedCredentials`，Auto-discovered credentials provided by the execution manager.
- `input_data`：`SendAuthenticatedWebRequestBlock.Input`，Input data for the HTTP block.
- `execution_context`：`ExecutionContext`，Execution context for the test.

返回值：`None`，This function does not return any value.

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Create auto-discovered credentials]
    B --> C[Setup mocks]
    C --> D[Create input data]
    D --> E[Execute with auto-discovered credentials]
    E --> F[Verify auto-discovered credentials were applied]
    F --> G[Verify response handling]
    G --> H[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_auto_discovered_credentials_flow(
    self,
    mock_requests_class,
    http_block,
    mock_response,
):
    # Create auto-discovered credentials
    auto_discovered_creds = HostScopedCredentials(
        provider="http",
        host="*.example.com",
        headers={
            "Authorization": SecretStr("Bearer auto-discovered-token"),
        },
        title="Auto-discovered Credentials",
    )

    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with empty credentials field (triggers auto-discovery)
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.GET,
        headers={},
        credentials=cast(
            HttpCredentials,
            {
                "id": "",  # Empty ID triggers auto-discovery in execution manager
                "provider": "http",
                "type": "host_scoped",
                "title": "",
            },
        ),
    )

    # Execute with auto-discovered credentials provided by execution manager
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=auto_discovered_creds,  # Execution manager found these
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify auto-discovered credentials were applied
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {"Authorization": "Bearer auto-discovered-token"}
    assert call_args.kwargs["headers"] == expected_headers

    # Verify response handling
    assert len(result) == 1
    assert result[0][0] == "response"
    assert result[0][1] == {"success": True, "data": "test"}
```



### test_multiple_header_credentials

This function tests that credentials with multiple headers are all applied to the HTTP request.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing HTTP requests.
- `http_block`：`SendAuthenticatedWebRequestBlock`，Instance of the HTTP block to be tested.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.
- `multi_header_creds`：`HostScopedCredentials`，Credentials with multiple headers.
- `input_data`：`SendAuthenticatedWebRequestBlock.Input`，Input data for the HTTP block.
- `execution_context`：`ExecutionContext`，Execution context for the test.

返回值：`None`，This function does not return any value.

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Create credentials with multiple headers]
    B --> C[Setup mocks]
    C --> D[Create input data]
    D --> E[Execute with multi-header credentials]
    E --> F[Verify all headers are included]
    F --> G[End]
```

#### 带注释源码

```python
@patch("backend.blocks.http.Requests")
async def test_multiple_header_credentials(
    self,
    mock_requests_class,
    http_block,
    mock_response,
):
    # Create credentials with multiple headers
    multi_header_creds = HostScopedCredentials(
        provider="http",
        host="api.example.com",
        headers={
            "Authorization": SecretStr("Bearer multi-token"),
            "X-API-Key": SecretStr("api-key-456"),
            "X-Client-ID": SecretStr("client-789"),
            "X-Custom-Header": SecretStr("custom-value"),
        },
        title="Multi-Header Credentials",
    )

    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with credentials containing multiple headers
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.GET,
        headers={"User-Agent": "test-agent"},
        credentials=cast(
            HttpCredentials,
            {
                "id": multi_header_creds.id,
                "provider": "http",
                "type": "host_scoped",
                "title": multi_header_creds.title,
            },
        ),
    )

    # Execute with multi-header credentials
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=multi_header_creds,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify all headers are included
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {
        "Authorization": "Bearer multi-token",
        "X-API-Key": "api-key-456",
        "X-Client-ID": "client-789",
        "X-Custom-Header": "custom-value",
        "User-Agent": "test-agent",
    }
    assert call_args.kwargs["headers"] == expected_headers
```



### test_credentials_with_complex_url_patterns

This function tests the credentials matching various URL patterns.

参数：

- `mock_requests_class`：`AsyncMock`，Mocked Requests class for testing HTTP requests.
- `http_block`：`SendAuthenticatedWebRequestBlock`，Instance of the HTTP block to be tested.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.
- `test_cases`：`list`，List of test cases with URL patterns and expected match results.

返回值：`None`，This function does not return any value.

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Create test cases]
    B --> C{Iterate over test cases}
    C -->|Should match| D[Create credentials]
    C -->|Should not match| E[Create credentials]
    D --> F[Create input data]
    E --> F
    F --> G[Execute with credentials]
    G --> H[Verify headers]
    H --> I[End]
    E --> J[Verify headers]
    J --> I
```

#### 带注释源码

```python
@patch("backend.blocks.http.Requests")
async def test_credentials_with_complex_url_patterns(
    self,
    mock_requests_class,
    http_block,
    mock_response,
):
    # Test cases for different URL patterns
    test_cases = [
        {
            "host_pattern": "api.example.com",
            "test_url": "https://api.example.com/v1/users",
            "should_match": True,
        },
        {
            "host_pattern": "*.example.com",
            "test_url": "https://api.example.com/v1/users",
            "should_match": True,
        },
        {
            "host_pattern": "*.example.com",
            "test_url": "https://subdomain.example.com/data",
            "should_match": True,
        },
        {
            "host_pattern": "api.example.com",
            "test_url": "https://api.different.com/data",
            "should_match": False,
        },
    ]

    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    for case in test_cases:
        # Reset mock for each test case
        mock_requests.reset_mock()

        # Create credentials for this test case
        test_creds = HostScopedCredentials(
            provider="http",
            host=case["host_pattern"],
            headers={
                "Authorization": SecretStr(f"Bearer {case['host_pattern']}-token"),
            },
            title=f"Credentials for {case['host_pattern']}",
        )

        input_data = SendAuthenticatedWebRequestBlock.Input(
            url=case["test_url"],
            method=HttpMethod.GET,
            headers={"User-Agent": "test-agent"},
            credentials=cast(
                HttpCredentials,
                {
                    "id": test_creds.id,
                    "provider": "http",
                    "type": "host_scoped",
                    "title": test_creds.title,
                },
            ),
        )

        # Execute with test credentials
        result = []
        async for output_name, output_data in http_block.run(
            input_data,
            credentials=test_creds,
            execution_context=make_test_context(),
        ):
            result.append((output_name, output_data))

        # Verify headers based on whether pattern should match
        mock_requests.request.assert_called_once()
        call_args = mock_requests.request.call_args
        headers = call_args.kwargs["headers"]

        if case["should_match"]:
            # Should include both user and credential headers
            expected_auth = f"Bearer {case['host_pattern']}-token"
            assert headers["Authorization"] == expected_auth
            assert headers["User-Agent"] == "test-agent"
        else:
            # Should only include user headers
            assert "Authorization" not in headers
            assert headers["User-Agent"] == "test-agent"
``` 



### make_test_context

Helper to create test ExecutionContext.

参数：

- `graph_exec_id`：`str`，The execution ID of the graph.
- `user_id`：`str`，The user ID for the execution context.

返回值：`ExecutionContext`，A test ExecutionContext object.

#### 流程图

```mermaid
graph TD
    A[make_test_context] --> B[Create ExecutionContext]
    B --> C[Return ExecutionContext]
```

#### 带注释源码

```python
def make_test_context(
    graph_exec_id: str = "test-exec-id",
    user_id: str = "test-user-id",
) -> ExecutionContext:
    """Helper to create test ExecutionContext."""
    return ExecutionContext(
        user_id=user_id,
        graph_exec_id=graph_exec_id,
    )
```



### `test_http_block_with_exact_host_match`

This function tests the HTTP block with exact host matching credentials.

参数：

- `mock_requests_class`：`MagicMock`，Mock of the Requests class used for making HTTP requests.
- `http_block`：`SendAuthenticatedWebRequestBlock`，Instance of the HTTP block to be tested.
- `exact_match_credentials`：`HostScopedCredentials`，Credentials for exact domain matching.
- `mock_response`：`MagicMock(spec=Response)`，Mock of the HTTP response object.

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with credentials]
    D --> E[Verify request headers]
    E --> F[Verify response handling]
    F --> G[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_http_block_with_exact_host_match(
    self,
    mock_requests_class,
    http_block,
    exact_match_credentials,
    mock_response,
):
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Prepare input data
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.GET,
        headers={"User-Agent": "test-agent"},
        credentials=cast(
            HttpCredentials,
            {
                "id": exact_match_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": exact_match_credentials.title,
            },
        ),
    )

    # Execute with credentials provided by execution manager
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=exact_match_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify request headers include both credential and user headers
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {
        "Authorization": "Bearer exact-match-token",
        "X-API-Key": "api-key-123",
        "User-Agent": "test-agent",
    }
    assert call_args.kwargs["headers"] == expected_headers

    # Verify response handling
    assert len(result) == 1
    assert result[0][0] == "response"
    assert result[0][1] == {"success": True, "data": "test"}
```



### `test_http_block_with_wildcard_host_match`

This function tests the HTTP block with wildcard host pattern matching credentials.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing
- `http_block`：`SendAuthenticatedWebRequestBlock`，HTTP block instance to be tested
- `wildcard_credentials`：`HostScopedCredentials`，Wildcard host pattern credentials
- `mock_response`：`MagicMock(spec=Response)`，Mocked successful HTTP response

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with wildcard credentials]
    D --> E[Verify wildcard matching works]
    E --> F[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_http_block_with_wildcard_host_match(
    self,
    mock_requests_class,
    http_block,
    wildcard_credentials,
    mock_response,
):
    """Test HTTP block with wildcard host pattern matching."""
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with subdomain that should match *.github.com
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.github.com/user",
        method=HttpMethod.GET,
        headers={},
        credentials=cast(
            HttpCredentials,
            {
                "id": wildcard_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": wildcard_credentials.title,
            },
        ),
    )

    # Execute with wildcard credentials
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=wildcard_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify wildcard matching works
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {"Authorization": "token ghp_wildcard123"}
    assert call_args.kwargs["headers"] == expected_headers
``` 



### `test_http_block_with_non_matching_credentials`

This function tests the HTTP block when the credentials provided do not match the target URL.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing HTTP requests.
- `http_block`：`SendAuthenticatedWebRequestBlock`，Instance of the HTTP block to be tested.
- `non_matching_credentials`：`HostScopedCredentials`，Credentials that do not match the target URL.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with non-matching credentials]
    D --> E[Verify request headers]
    E --> F[Verify response handling]
    F --> G[End]
```

#### 带注释源码

```python
@patch("backend.blocks.http.Requests")
async def test_http_block_with_non_matching_credentials(
    self,
    mock_requests_class,
    http_block,
    non_matching_credentials,
    mock_response,
):
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with URL that doesn't match the credentials
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.GET,
        headers={"User-Agent": "test-agent"},
        credentials=cast(
            HttpCredentials,
            {
                "id": non_matching_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": non_matching_credentials.title,
            },
        ),
    )

    # Execute with non-matching credentials
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=non_matching_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify only user headers are included (no credential headers)
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {"User-Agent": "test-agent"}
    assert call_args.kwargs["headers"] == expected_headers

    # Verify response handling
    assert len(result) == 1
    assert result[0][0] == "response"
    assert result[0][1] == {"success": True, "data": "test"}
```



### test_user_headers_override_credential_headers

This function tests that user-provided headers take precedence over credential headers when making an HTTP request.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class used to simulate HTTP requests.
- `http_block`：`SendAuthenticatedWebRequestBlock`，The HTTP block instance to be tested.
- `exact_match_credentials`：`HostScopedCredentials`，The credentials to be used for the test.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with conflicting headers]
    D --> E[Verify user headers take precedence]
    E --> F[End]
```

#### 带注释源码

```python
@patch("backend.blocks.http.Requests")
async def test_user_headers_override_credential_headers(
    self,
    mock_requests_class,
    http_block,
    exact_match_credentials,
    mock_response,
):
    """Test that user-provided headers take precedence over credential headers."""
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with user header that conflicts with credential header
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.POST,
        headers={
            "Authorization": "Bearer user-override-token",  # Should override
            "Content-Type": "application/json",  # Additional user header
        },
        credentials=cast(
            HttpCredentials,
            {
                "id": exact_match_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": exact_match_credentials.title,
            },
        ),
    )

    # Execute with conflicting headers
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=exact_match_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify user headers take precedence
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {
        "X-API-Key": "api-key-123",  # From credentials
        "Authorization": "Bearer user-override-token",  # User override
        "Content-Type": "application/json",  # User header
    }
    assert call_args.kwargs["headers"] == expected_headers
```



### `test_auto_discovered_credentials_flow`

This function tests the auto-discovery flow where the execution manager provides matching credentials.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing HTTP requests.
- `http_block`：`SendAuthenticatedWebRequestBlock`，HTTP block instance to be tested.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.
- `auto_discovered_creds`：`HostScopedCredentials`，Auto-discovered credentials for the test.
- `input_data`：`SendAuthenticatedWebRequestBlock.Input`，Input data for the HTTP block.
- `execution_context`：`ExecutionContext`，Execution context for the test.

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Create auto-discovered credentials]
    B --> C[Setup mocks]
    C --> D[Create input data with empty credentials]
    D --> E[Execute with auto-discovered credentials]
    E --> F[Verify auto-discovered credentials were applied]
    F --> G[Verify response handling]
    G --> H[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_auto_discovered_credentials_flow(
    self,
    mock_requests_class,
    http_block,
    mock_response,
):
    # Create auto-discovered credentials
    auto_discovered_creds = HostScopedCredentials(
        provider="http",
        host="*.example.com",
        headers={
            "Authorization": SecretStr("Bearer auto-discovered-token"),
        },
        title="Auto-discovered Credentials",
    )

    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with empty credentials field (triggers auto-discovery)
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.GET,
        headers={},
        credentials=cast(
            HttpCredentials,
            {
                "id": "",  # Empty ID triggers auto-discovery in execution manager
                "provider": "http",
                "type": "host_scoped",
                "title": "",
            },
        ),
    )

    # Execute with auto-discovered credentials provided by execution manager
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=auto_discovered_creds,  # Execution manager found these
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify auto-discovered credentials were applied
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {"Authorization": "Bearer auto-discovered-token"}
    assert call_args.kwargs["headers"] == expected_headers

    # Verify response handling
    assert len(result) == 1
    assert result[0][0] == "response"
    assert result[0][1] == {"success": True, "data": "test"}
``` 



### test_http_block_with_exact_host_match

Test HTTP block with exact host matching credentials.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing.
- `http_block`：`SendAuthenticatedWebRequestBlock`，HTTP block instance to be tested.
- `exact_match_credentials`：`HostScopedCredentials`，Host-scoped credentials for exact domain matching.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with credentials]
    D --> E[Verify request headers]
    E --> F[Verify response handling]
    F --> G[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_http_block_with_exact_host_match(
    self,
    mock_requests_class,
    http_block,
    exact_match_credentials,
    mock_response,
):
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Prepare input data
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.GET,
        headers={"User-Agent": "test-agent"},
        credentials=cast(
            HttpCredentials,
            {
                "id": exact_match_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": exact_match_credentials.title,
            },
        ),
    )

    # Execute with credentials provided by execution manager
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=exact_match_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify request headers include both credential and user headers
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {
        "Authorization": "Bearer exact-match-token",
        "X-API-Key": "api-key-123",
        "User-Agent": "test-agent",
    }
    assert call_args.kwargs["headers"] == expected_headers

    # Verify response handling
    assert len(result) == 1
    assert result[0][0] == "response"
    assert result[0][1] == {"success": True, "data": "test"}
```

### test_http_block_with_wildcard_host_match

Test HTTP block with wildcard host pattern matching.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing.
- `http_block`：`SendAuthenticatedWebRequestBlock`，HTTP block instance to be tested.
- `wildcard_credentials`：`HostScopedCredentials`，Host-scoped credentials with wildcard pattern.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with wildcard credentials]
    D --> E[Verify wildcard matching works]
    E --> F[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_http_block_with_wildcard_host_match(
    self,
    mock_requests_class,
    http_block,
    wildcard_credentials,
    mock_response,
):
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with subdomain that should match *.github.com
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.github.com/user",
        method=HttpMethod.GET,
        headers={},
        credentials=cast(
            HttpCredentials,
            {
                "id": wildcard_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": wildcard_credentials.title,
            },
        ),
    )

    # Execute with wildcard credentials
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=wildcard_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify wildcard matching works
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {"Authorization": "token ghp_wildcard123"}
    assert call_args.kwargs["headers"] == expected_headers
```

### test_http_block_with_non_matching_credentials

Test HTTP block when credentials don't match the target URL.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing.
- `http_block`：`SendAuthenticatedWebRequestBlock`，HTTP block instance to be tested.
- `non_matching_credentials`：`HostScopedCredentials`，Credentials that don't match test URLs.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with non-matching credentials]
    D --> E[Verify only user headers are included]
    E --> F[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_http_block_with_non_matching_credentials(
    self,
    mock_requests_class,
    http_block,
    non_matching_credentials,
    mock_response,
):
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with URL that doesn't match the credentials
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.GET,
        headers={"User-Agent": "test-agent"},
        credentials=cast(
            HttpCredentials,
            {
                "id": non_matching_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": non_matching_credentials.title,
            },
        ),
    )

    # Execute with non-matching credentials
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=non_matching_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify only user headers are included (no credential headers)
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {"User-Agent": "test-agent"}
    assert call_args.kwargs["headers"] == expected_headers
```

### test_user_headers_override_credential_headers

Test that user-provided headers take precedence over credential headers.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing.
- `http_block`：`SendAuthenticatedWebRequestBlock`，HTTP block instance to be tested.
- `exact_match_credentials`：`HostScopedCredentials`，Host-scoped credentials for exact domain matching.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Prepare input data]
    C --> D[Execute with conflicting headers]
    D --> E[Verify user headers take precedence]
    E --> F[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_user_headers_override_credential_headers(
    self,
    mock_requests_class,
    http_block,
    exact_match_credentials,
    mock_response,
):
    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with user header that conflicts with credential header
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.POST,
        headers={
            "Authorization": "Bearer user-override-token",  # Should override
            "Content-Type": "application/json",  # Additional user header
        },
        credentials=cast(
            HttpCredentials,
            {
                "id": exact_match_credentials.id,
                "provider": "http",
                "type": "host_scoped",
                "title": exact_match_credentials.title,
            },
        ),
    )

    # Execute with conflicting headers
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=exact_match_credentials,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify user headers take precedence
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {
        "X-API-Key": "api-key-123",  # From credentials
        "Authorization": "Bearer user-override-token",  # User override
        "Content-Type": "application/json",  # User header
    }
    assert call_args.kwargs["headers"] == expected_headers
```

### test_auto_discovered_credentials_flow

Test the auto-discovery flow where execution manager provides matching credentials.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing.
- `http_block`：`SendAuthenticatedWebRequestBlock`，HTTP block instance to be tested.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Create auto-discovered credentials]
    B --> C[Setup mocks]
    C --> D[Prepare input data]
    D --> E[Execute with auto-discovered credentials]
    E --> F[Verify auto-discovered credentials were applied]
    F --> G[Verify response handling]
    G --> H[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_auto_discovered_credentials_flow(
    self,
    mock_requests_class,
    http_block,
    mock_response,
):
    # Create auto-discovered credentials
    auto_discovered_creds = HostScopedCredentials(
        provider="http",
        host="*.example.com",
        headers={
            "Authorization": SecretStr("Bearer auto-discovered-token"),
        },
        title="Auto-discovered Credentials",
    )

    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with empty credentials field (triggers auto-discovery)
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.GET,
        headers={},
        credentials=cast(
            HttpCredentials,
            {
                "id": "",  # Empty ID triggers auto-discovery in execution manager
                "provider": "http",
                "type": "host_scoped",
                "title": "",
            },
        ),
    )

    # Execute with auto-discovered credentials provided by execution manager
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=auto_discovered_creds,  # Execution manager found these
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify auto-discovered credentials were applied
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {"Authorization": "Bearer auto-discovered-token"}
    assert call_args.kwargs["headers"] == expected_headers

    # Verify response handling
    assert len(result) == 1
    assert result[0][0] == "response"
    assert result[0][1] == {"success": True, "data": "test"}
```

### test_multiple_header_credentials

Test credentials with multiple headers are all applied.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing.
- `http_block`：`SendAuthenticatedWebRequestBlock`，HTTP block instance to be tested.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Create credentials with multiple headers]
    B --> C[Setup mocks]
    C --> D[Prepare input data]
    D --> E[Execute with multi-header credentials]
    E --> F[Verify all headers are included]
    F --> G[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_multiple_header_credentials(
    self,
    mock_requests_class,
    http_block,
    mock_response,
):
    # Create credentials with multiple headers
    multi_header_creds = HostScopedCredentials(
        provider="http",
        host="api.example.com",
        headers={
            "Authorization": SecretStr("Bearer multi-token"),
            "X-API-Key": SecretStr("api-key-456"),
            "X-Client-ID": SecretStr("client-789"),
            "X-Custom-Header": SecretStr("custom-value"),
        },
        title="Multi-Header Credentials",
    )

    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    # Test with credentials containing multiple headers
    input_data = SendAuthenticatedWebRequestBlock.Input(
        url="https://api.example.com/data",
        method=HttpMethod.GET,
        headers={"User-Agent": "test-agent"},
        credentials=cast(
            HttpCredentials,
            {
                "id": multi_header_creds.id,
                "provider": "http",
                "type": "host_scoped",
                "title": multi_header_creds.title,
            },
        ),
    )

    # Execute with multi-header credentials
    result = []
    async for output_name, output_data in http_block.run(
        input_data,
        credentials=multi_header_creds,
        execution_context=make_test_context(),
    ):
        result.append((output_name, output_data))

    # Verify all headers are included
    mock_requests.request.assert_called_once()
    call_args = mock_requests.request.call_args
    expected_headers = {
        "Authorization": "Bearer multi-token",
        "X-API-Key": "api-key-456",
        "X-Client-ID": "client-789",
        "X-Custom-Header": "custom-value",
        "User-Agent": "test-agent",
    }
    assert call_args.kwargs["headers"] == expected_headers
```

### test_credentials_with_complex_url_patterns

Test credentials matching various URL patterns.

参数：

- `mock_requests_class`：`MagicMock`，Mocked Requests class for testing.
- `http_block`：`SendAuthenticatedWebRequestBlock`，HTTP block instance to be tested.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.

返回值：无

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Setup mocks]
    B --> C[Define test cases]
    C --> D[Loop through test cases]
    D --> E[Create credentials for test case]
    E --> F[Prepare input data]
    F --> G[Execute with test credentials]
    G --> H[Verify headers based on pattern]
    H --> I[End loop]
    I --> J[End]
```

#### 带注释源码

```python
@pytest.mark.asyncio
@patch("backend.blocks.http.Requests")
async def test_credentials_with_complex_url_patterns(
    self,
    mock_requests_class,
    http_block,
    mock_response,
):
    # Test cases for different URL patterns
    test_cases = [
        {
            "host_pattern": "api.example.com",
            "test_url": "https://api.example.com/v1/users",
            "should_match": True,
        },
        {
            "host_pattern": "*.example.com",
            "test_url": "https://api.example.com/v1/users",
            "should_match": True,
        },
        {
            "host_pattern": "*.example.com",
            "test_url": "https://subdomain.example.com/data",
            "should_match": True,
        },
        {
            "host_pattern": "api.example.com",
            "test_url": "https://api.different.com/data",
            "should_match": False,
        },
    ]

    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    for case in test_cases:
        # Reset mock for each test case
        mock_requests.reset_mock()

        # Create credentials for this test case
        test_creds = HostScopedCredentials(
            provider="http",
            host=case["host_pattern"],
            headers={
                "Authorization": SecretStr(f"Bearer {case['host_pattern']}-token"),
            },
            title=f"Credentials for {case['host_pattern']}",
        )

        input_data = SendAuthenticatedWebRequestBlock.Input(
            url=case["test_url"],
            method=HttpMethod.GET,
            headers={"User-Agent": "test-agent"},
            credentials=cast(
                HttpCredentials,
                {
                    "id": test_creds.id,
                    "provider": "http",
                    "type": "host_scoped",
                    "title": test_creds.title,
                },
            ),
        )

        # Execute with test credentials
        result = []
        async for output_name, output_data in http_block.run(
            input_data,
            credentials=test_creds,
            execution_context=make_test_context(),
        ):
            result.append((output_name, output_data))

        # Verify headers based on whether pattern should match
        mock_requests.request.assert_called_once()
        call_args = mock_requests.request.call_args
        headers = call_args.kwargs["headers"]

        if case["should_match"]:
            # Should include both user and credential headers
            expected_auth = f"Bearer {case['host_pattern']}-token"
            assert headers["Authorization"] == expected_auth
            assert headers["User-Agent"] == "test-agent"
        else:
            # Should only include user headers
            assert "Authorization" not in headers
            assert headers["User-Agent"] == "test-agent"
```


### test_credentials_with_complex_url_patterns

This function tests the credentials matching various URL patterns.

参数：

- `mock_requests_class`：`AsyncMock`，Mocked Requests class for testing HTTP requests.
- `http_block`：`SendAuthenticatedWebRequestBlock`，HTTP block instance for sending authenticated web requests.
- `mock_response`：`MagicMock(spec=Response)`，Mocked HTTP response object.
- `test_cases`：`list`，List of test cases with URL patterns and expected match results.

返回值：`None`，This function does not return any value.

#### 流程图

```mermaid
graph TD
    A[Start] --> B[Create credentials for test case]
    B --> C[Create input data]
    C --> D[Execute HTTP block]
    D --> E[Verify headers based on match result]
    E --> F[End]
```

#### 带注释源码

```python
@patch("backend.blocks.http.Requests")
async def test_credentials_with_complex_url_patterns(
    self,
    mock_requests_class,
    http_block,
    mock_response,
):
    # Test cases for different URL patterns
    test_cases = [
        {
            "host_pattern": "api.example.com",
            "test_url": "https://api.example.com/v1/users",
            "should_match": True,
        },
        {
            "host_pattern": "*.example.com",
            "test_url": "https://api.example.com/v1/users",
            "should_match": True,
        },
        {
            "host_pattern": "*.example.com",
            "test_url": "https://subdomain.example.com/data",
            "should_match": True,
        },
        {
            "host_pattern": "api.example.com",
            "test_url": "https://api.different.com/data",
            "should_match": False,
        },
    ]

    # Setup mocks
    mock_requests = AsyncMock()
    mock_requests.request.return_value = mock_response
    mock_requests_class.return_value = mock_requests

    for case in test_cases:
        # Reset mock for each test case
        mock_requests.reset_mock()

        # Create credentials for this test case
        test_creds = HostScopedCredentials(
            provider="http",
            host=case["host_pattern"],
            headers={
                "Authorization": SecretStr(f"Bearer {case['host_pattern']}-token"),
            },
            title=f"Credentials for {case['host_pattern']}",
        )

        input_data = SendAuthenticatedWebRequestBlock.Input(
            url=case["test_url"],
            method=HttpMethod.GET,
            headers={"User-Agent": "test-agent"},
            credentials=cast(
                HttpCredentials,
                {
                    "id": test_creds.id,
                    "provider": "http",
                    "type": "host_scoped",
                    "title": test_creds.title,
                },
            ),
        )

        # Execute with test credentials
        result = []
        async for output_name, output_data in http_block.run(
            input_data,
            credentials=test_creds,
            execution_context=make_test_context(),
        ):
            result.append((output_name, output_data))

        # Verify headers based on whether pattern should match
        mock_requests.request.assert_called_once()
        call_args = mock_requests.request.call_args
        headers = call_args.kwargs["headers"]

        if case["should_match"]:
            # Should include both user and credential headers
            expected_auth = f"Bearer {case['host_pattern']}-token"
            assert headers["Authorization"] == expected_auth
            assert headers["User-Agent"] == "test-agent"
        else:
            # Should only include user headers
            assert "Authorization" not in headers
            assert headers["User-Agent"] == "test-agent"
``` 


## 关键组件


### 张量索引与惰性加载

张量索引与惰性加载是代码中用于高效处理大型数据集的关键组件，它允许在需要时才计算数据，从而减少内存消耗和提高性能。

### 反量化支持

反量化支持是代码中用于处理量化数据的关键组件，它允许在量化过程中进行反向操作，以便在需要时恢复原始数据。

### 量化策略

量化策略是代码中用于优化模型性能的关键组件，它通过减少模型中使用的精度来减少模型大小和计算需求，从而提高模型在资源受限环境中的运行效率。



## 问题及建议


### 已知问题

-   **代码重复性**：在多个测试用例中，对于`HostScopedCredentials`的创建和配置存在重复代码。这可能导致维护成本增加，并且如果需要修改这些配置，需要在多个地方进行更改。
-   **测试用例的覆盖范围**：虽然测试用例涵盖了多种场景，但可能缺少一些边缘情况，例如URL编码、特殊字符处理等。
-   **异常处理**：代码中没有显示异常处理逻辑，如果请求失败或响应解析出错，可能会导致测试失败，但没有提供明确的错误信息。

### 优化建议

-   **提取公共代码**：将创建和配置`HostScopedCredentials`的代码提取到一个单独的函数中，减少重复代码，并提高代码的可维护性。
-   **增加边缘情况测试**：为测试用例增加更多边缘情况的测试，例如URL编码、特殊字符处理、网络错误等，以确保代码的健壮性。
-   **添加异常处理**：在请求和响应处理中添加异常处理逻辑，确保在出现错误时能够捕获异常并给出清晰的错误信息，提高代码的可用性。
-   **代码注释**：在代码中添加必要的注释，解释代码的功能和逻辑，提高代码的可读性。
-   **代码格式化**：确保代码格式一致，使用代码格式化工具进行格式化，提高代码的可读性。


## 其它


### 设计目标与约束

- 设计目标：
  - 确保HTTP请求块能够安全、高效地发送认证请求。
  - 支持多种认证类型，包括主机范围认证。
  - 确保用户自定义头信息能够覆盖凭证头信息。
  - 提供自动发现凭证的功能，以简化配置。
- 约束：
  - 必须使用现有的认证机制。
  - 代码必须符合安全最佳实践，如避免明文存储敏感信息。
  - 代码必须易于维护和扩展。

### 错误处理与异常设计

- 错误处理：
  - 对于HTTP请求失败，应捕获异常并返回错误信息。
  - 对于凭证匹配失败，应返回相应的错误信息。
  - 对于配置错误，应提供详细的错误描述。
- 异常设计：
  - 定义自定义异常类，以处理特定错误情况。
  - 使用try-except块捕获和处理异常。

### 数据流与状态机

- 数据流：
  - 输入数据包括URL、方法、头信息和凭证。
  - 输出数据包括响应状态、响应数据和错误信息。
- 状态机：
  - 状态包括初始化、请求发送、响应接收和错误处理。

### 外部依赖与接口契约

- 外部依赖：
  - `Requests`库用于发送HTTP请求。
  - `pydantic`库用于数据验证。
- 接口契约：
  - `SendAuthenticatedWebRequestBlock`类定义了输入和输出接口。
  - `HostScopedCredentials`类定义了凭证接口。


    