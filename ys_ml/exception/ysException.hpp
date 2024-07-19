#include <exception>

class YSException : public std::exception {
    public:
        YSException(const char* message);
        const char* what();

    private:
        const char* m_message;
};