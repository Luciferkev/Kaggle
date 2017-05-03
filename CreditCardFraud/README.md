Credit Card Fraud Detection Problem From kaggle <br/>
visit : https://www.kaggle.com/dalpozz/creditcardfraud
<br/><br/>

<b><i>Flow of the algorithm:</i></b>
      <ul>
        <li>Statistical Analysis</li>
        <li>Features Analysis</li>
        <li>Tensorflow Graph Development</li>
      </ul>


<b>To predict any value from the dataset:</b>
      <ul>
            <li>inputY is hot-mapped as normal/fraud transactions:</li><br/>
                  <i>| 0 | 1 |(Normal Transaction) || | 1 | 0 |(Fraud Transaction)</i><br/>
                  <table>
                      <tr>
                          <td> Fraud </td>
                          <td> Normal </td>
                      </tr>     
                      <tr>
                          <td>   0-1  </td>
                          <td>   0-1  </td>
                      </tr>
                  </table>
      </ul>
