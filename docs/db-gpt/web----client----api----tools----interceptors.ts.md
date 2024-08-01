# `.\DB-GPT-src\web\client\api\tools\interceptors.ts`

```py
/**
 * Response processing function that handles API responses and errors.
 *
 * @param promise Promise representing the API request
 * @param ignoreCodes Optional array of error codes to ignore or '*' to ignore all
 * @returns Promise resolving to a tuple with success or failure details
 */
export const apiInterceptors = <T = any, D = any>(promise: Promise<ApiResponse<T, D>>, ignoreCodes?: '*' | (number | string)[]) => {
    // Handle the promise resolving with a successful response
    return promise
        .then<SuccessTuple<T, D>>((response) => {
            const { data } = response;
            // If response data is missing, throw a network error
            if (!data) {
                throw new Error('Network Error!');
            }
            // If response indicates failure and not ignored, show notification
            if (!data.success) {
                if (ignoreCodes === '*' || (data.err_code && ignoreCodes && ignoreCodes.includes(data.err_code))) {
                    // Return tuple with no error but with data and response
                    return [null, data.data, data, response];
                } else {
                    // Show notification for the request error
                    notification.error({
                        message: `Request error`,
                        description: data?.err_msg ?? 'The interface is abnormal. Please try again later',
                    });
                }
            }
            // Return tuple with no error but with data and response
            return [null, data.data, data, response];
        })
        .catch<FailedTuple<T, D>>((err: Error | AxiosError<T, D>) => {
            let errMessage = err.message;
            // Handle AxiosError to extract detailed error message
            if (err instanceof AxiosError) {
                try {
                    const { err_msg } = JSON.parse(err.request.response) as ResponseType<null>;
                    err_msg && (errMessage = err_msg);
                } catch (e) {}
            }
            // Show notification for the request error
            notification.error({
                message: `Request error`,
                description: errMessage,
            });
            // Return tuple with error details
            return [err, null, null, null];
        });
};
```