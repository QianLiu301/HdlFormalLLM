// traffic_light — MUTANT (bug07_outputs_registered.v) — lamp outputs registered: every indication is one cycle late
module traffic_light #(
    parameter G_TICKS = 8,
    parameter Y_TICKS = 3,
    parameter R_TICKS = 6
) (
    input  wire clk,
    input  wire rst_n,
    output wire red,
    output wire yellow,
    output wire green
);
    localparam S_RED = 2'd0, S_GREEN = 2'd1, S_YELLOW = 2'd2;
    reg [1:0] state;
    reg [7:0] cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_RED;
            cnt   <= 8'd0;
        end else begin
            case (state)
                S_RED:    if (cnt == R_TICKS - 1) begin state <= S_GREEN;  cnt <= 8'd0; end
                          else cnt <= cnt + 8'd1;
                S_GREEN:  if (cnt == G_TICKS - 1) begin state <= S_YELLOW; cnt <= 8'd0; end
                          else cnt <= cnt + 8'd1;
                S_YELLOW: if (cnt == Y_TICKS - 1) begin state <= S_RED;    cnt <= 8'd0; end
                          else cnt <= cnt + 8'd1;
                default:  begin state <= S_RED; cnt <= 8'd0; end
            endcase
        end
    end

    reg red_r, yellow_r, green_r;          // BUG: outputs delayed one cycle
    always @(posedge clk) begin
        red_r    <= (state == S_RED);
        yellow_r <= (state == S_YELLOW);
        green_r  <= (state == S_GREEN);
    end
    assign red    = red_r;
    assign yellow = yellow_r;
    assign green  = green_r;
endmodule
