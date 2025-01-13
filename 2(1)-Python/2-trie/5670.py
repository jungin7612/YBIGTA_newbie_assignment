from lib import Trie
import sys


"""
TODO:
- 일단 Trie부터 구현하기
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer = 0
    cnt = 0

    for element in query_seq:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        
        new_index = -1 # 구현하세요!
        for k in trie[pointer].children:
            if  trie[k].body == element:
                new_index = k
                break
      
        pointer = new_index

    return cnt + int(len(trie[0].children) == 1)


def main() -> None:
    # 구현하세요!
    while True:
        try:
  
            trie: Trie = Trie()
            N = int(sys.stdin.readline().strip())
            str_list = []
            for _ in range(N):
                str_list.append(sys.stdin.readline().strip())

            s = 0
            for word in str_list:
                trie.push(word)
            for word in str_list:
                s += count(trie,word)

            result = round(s / N, 2)
            print(format(result,".2f"))
        except:
            break
        
    


if __name__ == "__main__":
    main()